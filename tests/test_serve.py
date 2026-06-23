import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from receipt_print.serve import (
    PrintQueueFull,
    PrintService,
    PrintServiceStopped,
    PrintTimeout,
    _strip_speed_env,
    make_server,
)

FIXTURE = Path(__file__).parent / "fixtures" / "receipt_sample.bin"


@pytest.fixture
def service():
    svc = PrintService()
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


@pytest.fixture
def http_server(monkeypatch):
    """A real ThreadingHTTPServer bound to 127.0.0.1:0 with a recording printer."""
    recorder = {"writes": [], "cut_called": False, "closed": 0}

    class Recorder:
        def _raw(self, data):
            recorder["writes"].append(data)

        def cut(self):
            recorder["cut_called"] = True

        def close(self):
            recorder["closed"] += 1

    monkeypatch.setattr(
        "receipt_print.printer.connect_printer", lambda: Recorder()
    )
    server, svc = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield port, recorder
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def _post(port, body, path="/v1/print/raw", content_length=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/octet-stream"}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    status = resp.status
    ctype = resp.getheader("Content-Type")
    data = resp.read()
    conn.close()
    return status, ctype, data


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    status = resp.status
    ctype = resp.getheader("Content-Type")
    data = resp.read()
    conn.close()
    return status, ctype, data


def test_byte_passthrough_single_write(http_server, monkeypatch):
    seen = {}

    def recorder(data, cut=False):
        seen["data"] = data
        seen["cut"] = cut

    monkeypatch.setattr("receipt_print.serve.print_raw_bytes", recorder)
    port, _ = http_server
    payload = b"\x1b@hello\x1dV\x00"
    status, ctype, data = _post(port, payload)
    assert status == 200
    assert ctype == "application/json"
    assert json.loads(data) == {"ok": True, "bytes": len(payload)}
    assert seen["data"] == payload
    assert seen["cut"] is False


def test_no_extra_cut_via_real_print_raw_bytes(http_server):
    port, recorder = http_server
    payload = bytes(range(256)) + b"\x1d\x56\x00"
    status, ctype, data = _post(port, payload)
    assert status == 200
    assert recorder["writes"] == [payload]
    assert recorder["cut_called"] is False
    assert recorder["closed"] == 1


def test_heterogeneous_opaque_jobs_serialize(service):
    order = []

    def raw_like():
        order.append("raw")
        return 7

    def marker():
        order.append("marker")
        return "marked"

    def number():
        order.append("number")
        return 42

    assert service.submit(raw_like) == 7
    assert service.submit(marker) == "marked"
    assert service.submit(number) == 42
    assert order == ["raw", "marker", "number"]


def test_serial_execution_no_overlap(service):
    intervals = []
    lock = threading.Lock()

    def job(i):
        start = time.monotonic()
        time.sleep(0.02)
        end = time.monotonic()
        with lock:
            intervals.append((start, end))

    threads = [
        threading.Thread(target=lambda i=i: service.submit(lambda i=i: job(i)))
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    intervals.sort()
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
        assert next_start >= prev_end - 1e-3


def test_systemexit_in_job_does_not_kill_worker(service):
    def boom():
        sys.exit(1)

    with pytest.raises(SystemExit):
        service.submit(boom)

    assert service.submit(lambda: 42) == 42
    assert service.health()["worker_alive"] is True


def test_job_exception_reraised(service):
    def boom():
        raise RuntimeError("printer on fire")

    with pytest.raises(RuntimeError, match="printer on fire"):
        service.submit(boom)
    assert service.submit(lambda: "ok") == "ok"


def test_timeout_orphan_not_printed():
    svc = PrintService()
    svc.start()
    try:
        release = threading.Event()
        first_done = threading.Event()
        side_effects = []

        def blocker():
            side_effects.append("blocker")
            release.wait(timeout=5)

        def orphan():
            side_effects.append("orphan")

        def run_blocker():
            svc.submit(blocker)
            first_done.set()

        t = threading.Thread(target=run_blocker)
        t.start()
        while "blocker" not in side_effects:
            time.sleep(0.001)

        with pytest.raises(PrintTimeout):
            svc.submit(orphan, timeout=0.05)

        release.set()
        first_done.wait(timeout=5)
        t.join(timeout=5)
        time.sleep(0.05)
        assert "orphan" not in side_effects
    finally:
        release.set()
        svc.shutdown()


def test_queue_full():
    svc = PrintService(queue_max=1)
    svc.start()
    try:
        release = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=5)

        threading.Thread(target=lambda: svc.submit(blocker)).start()
        started.wait(timeout=5)

        threading.Thread(
            target=lambda: svc.submit(lambda: None, enqueue_timeout=5)
        ).start()
        time.sleep(0.05)

        with pytest.raises(PrintQueueFull):
            svc.submit(lambda: None, enqueue_timeout=0.05)

        release.set()
    finally:
        release.set()
        svc.shutdown()


def test_clean_shutdown_drains_inflight():
    svc = PrintService()
    svc.start()
    ran = []
    svc.submit(lambda: ran.append("job"))
    svc.shutdown()
    assert ran == ["job"]
    assert svc.health()["worker_alive"] is False
    with pytest.raises(PrintServiceStopped):
        svc.submit(lambda: None)


def test_http_empty_body_is_text_plain(http_server):
    port, _ = http_server
    status, ctype, _ = _post(port, b"", content_length=0)
    assert status == 400
    assert ctype.startswith("text/plain")


def test_http_oversized_is_text_plain(monkeypatch):
    recorder = {"writes": []}

    class Recorder:
        def _raw(self, data):
            recorder["writes"].append(data)

        def close(self):
            pass

    monkeypatch.setattr(
        "receipt_print.printer.connect_printer", lambda: Recorder()
    )
    server, svc = make_server("127.0.0.1", 0, max_bytes=16)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, ctype, _ = _post(port, b"x" * 64)
        assert status == 413
        assert ctype.startswith("text/plain")
        assert recorder["writes"] == []
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def test_http_job_error_is_text_plain(monkeypatch):
    def boom(data, cut=False):
        raise RuntimeError("device unplugged")

    monkeypatch.setattr("receipt_print.serve.print_raw_bytes", boom)
    server, svc = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, ctype, data = _post(port, b"hello")
        assert status == 502
        assert ctype.startswith("text/plain")
        assert b"device unplugged" in data
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def test_http_systemexit_in_job_surfaces_then_recovers(monkeypatch):
    state = {"boom": True}

    def maybe_boom(data, cut=False):
        if state["boom"]:
            sys.exit(1)

    monkeypatch.setattr("receipt_print.serve.print_raw_bytes", maybe_boom)
    server, svc = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, ctype, data = _post(port, b"hello")
        assert status == 502
        assert ctype.startswith("text/plain")
        state["boom"] = False
        status2, ctype2, data2 = _post(port, b"world")
        assert status2 == 200
        assert json.loads(data2) == {"ok": True, "bytes": 5}
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def test_healthz_reports_worker_alive(http_server):
    port, _ = http_server
    status, ctype, data = _get(port, "/healthz")
    assert status == 200
    assert ctype == "application/json"
    body = json.loads(data)
    assert body["worker_alive"] is True
    assert "queue" in body


def test_unknown_path_404_text_plain(http_server):
    port, _ = http_server
    status, ctype, _ = _get(port, "/nope")
    assert status == 404
    assert ctype.startswith("text/plain")


def test_roundtrip_real_receipt_fixture(http_server):
    port, recorder = http_server
    sample = FIXTURE.read_bytes()
    status, ctype, data = _post(port, sample)
    assert status == 200
    assert ctype == "application/json"
    assert json.loads(data) == {"ok": True, "bytes": len(sample)}
    assert recorder["writes"] == [sample]
    assert len(recorder["writes"]) == 1
    assert recorder["cut_called"] is False


def test_alt_raw_path_accepted(http_server):
    port, recorder = http_server
    status, _, _ = _post(port, b"abc", path="/print/raw")
    assert status == 200
    assert recorder["writes"] == [b"abc"]


def test_strip_speed_env_is_structural(monkeypatch):
    monkeypatch.setenv("RP_SPEED_OVERRIDE", "50")
    monkeypatch.setenv("RP_SPEED", "50")
    _strip_speed_env()
    assert os.getenv("RP_SPEED_OVERRIDE") is None
    assert os.getenv("RP_SPEED") is None
    assert os.getenv("RP_HOST") is None
