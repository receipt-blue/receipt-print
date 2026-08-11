import http.client
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import receipt_print.serve as serve_module
from receipt_print.journal import PrintJournal
from receipt_print.printer import PrinterUnavailableError

from receipt_print.serve import (
    PrintQueueFull,
    PrintService,
    PrintServiceStopped,
    PrintTimeout,
    _strip_speed_env,
    make_server as make_production_server,
)

FIXTURE = Path(__file__).parent / "fixtures" / "receipt_sample.bin"


def make_server(*args, **kwargs):
    kwargs.setdefault(
        "executor", lambda data: serve_module.print_raw_bytes(data, cut=False)
    )
    return make_production_server(*args, **kwargs)


@pytest.fixture
def service():
    svc = PrintService()
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


@pytest.fixture
def http_server(monkeypatch, tmp_path):
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
    monkeypatch.setattr("receipt_print.serve.printer_device_available", lambda: True)
    server, svc = make_server(
        "127.0.0.1",
        0,
        journal=PrintJournal(str(tmp_path / "jobs.sqlite3")),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield port, recorder
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def _post(
    port,
    body,
    path="/v1/print/raw",
    content_length=None,
    content_type="application/octet-stream",
    job_id=None,
):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": content_type}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    if job_id is not None:
        headers["X-Receipt-Print-Job-Id"] = job_id
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


def _read_until(client, marker):
    response = b""
    while marker not in response:
        chunk = client.recv(4096)
        if not chunk:
            break
        response += chunk
    return response


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
    assert json.loads(data) == {"success": True, "bytes": len(payload)}
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


def test_job_exception_reraised(service, monkeypatch):
    monkeypatch.setattr("receipt_print.serve.printer_device_available", lambda: True)

    def boom():
        raise RuntimeError("printer on fire")

    with pytest.raises(RuntimeError, match="printer on fire"):
        service.submit(boom)
    failed = service.health()
    assert failed["ready"] is False
    assert failed["status"] == "error"
    assert failed["last_error"] == "RuntimeError: printer on fire"
    assert failed["last_error_at"] is not None
    assert service.submit(lambda: "ok") == "ok"
    recovered = service.health()
    assert recovered["ready"] is True
    assert recovered["status"] == "ready"
    assert recovered["last_error"] is None
    assert recovered["last_success_at"] is not None


def test_new_device_identity_clears_delivery_error(service, monkeypatch, tmp_path):
    device = tmp_path / "printer"
    device.write_bytes(b"")
    monkeypatch.setenv("RP_DEVICE", str(device))
    monkeypatch.setattr("receipt_print.serve.printer_device_available", lambda: True)

    with pytest.raises(OSError, match="stalled"):
        service.submit(lambda: (_ for _ in ()).throw(OSError("stalled")))
    assert service.health()["status"] == "error"

    device.unlink()
    device.write_bytes(b"")
    recovered = service.health()
    assert recovered["ready"] is True
    assert recovered["status"] == "ready"
    assert recovered["last_error"] is None


def test_timeout_orphan_not_printed():
    svc = PrintService()
    svc.start()
    release = threading.Event()
    try:
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
    release = threading.Event()
    try:
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


def test_shutdown_deadline_cancels_queued_jobs():
    svc = PrintService(queue_max=1, drain_grace=0.05)
    svc.start()
    release = threading.Event()
    started = threading.Event()
    result = {}

    def blocker():
        started.set()
        release.wait(timeout=5)

    def submit_queued():
        try:
            svc.submit(lambda: None, enqueue_timeout=5)
        except BaseException as error:
            result["error"] = error

    try:
        threading.Thread(target=lambda: svc.submit(blocker)).start()
        assert started.wait(timeout=5)
        queued = threading.Thread(target=submit_queued)
        queued.start()
        time.sleep(0.02)

        began = time.monotonic()
        svc.shutdown(grace=0.05)
        elapsed = time.monotonic() - began

        assert elapsed < 0.2
        queued.join(timeout=1)
        assert isinstance(result.get("error"), PrintServiceStopped)
    finally:
        release.set()
        svc.shutdown(grace=1)


def test_http_empty_body_is_text_plain(http_server):
    port, _ = http_server
    status, ctype, _ = _post(port, b"", content_length=0)
    assert status == 400
    assert ctype.startswith("text/plain")


@pytest.mark.parametrize(
    ("content_length", "expected_status"),
    [("-1", 400), ("not-a-number", 400), ("9" * 5000, 413)],
)
def test_http_rejects_malformed_content_length(
    http_server, content_length, expected_status
):
    port, recorder = http_server
    status, ctype, _ = _post(port, b"hello", content_length=content_length)
    assert status == expected_status
    assert ctype.startswith("text/plain")
    assert recorder["writes"] == []


def test_http_rejects_wrong_content_type(http_server):
    port, recorder = http_server
    status, ctype, _ = _post(port, b"hello", content_type="text/plain")
    assert status == 415
    assert ctype.startswith("text/plain")
    assert recorder["writes"] == []


def test_http_stalled_body_times_out_and_tears_down(monkeypatch):
    recorder = {"writes": []}

    class Recorder:
        def _raw(self, data):
            recorder["writes"].append(data)

        def close(self):
            pass

    monkeypatch.setattr(
        "receipt_print.printer.connect_printer", lambda: Recorder()
    )
    server, svc = make_server("127.0.0.1", 0, request_timeout=0.05)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=1)
    client.sendall(
        b"POST /v1/print/raw HTTP/1.0\r\n"
        b"Content-Type: application/octet-stream\r\n"
        b"Content-Length: 5\r\n\r\n"
    )
    try:
        response = _read_until(client, b"request body timed out")
        assert b"408 Request Timeout" in response
        assert b"request body timed out" in response
        assert recorder["writes"] == []
    finally:
        client.close()
        began = time.monotonic()
        server.shutdown()
        server.server_close()
        svc.shutdown(grace=0.2)
        assert time.monotonic() - began < 0.5


def test_http_short_body_is_bad_request(http_server):
    port, recorder = http_server
    client = socket.create_connection(("127.0.0.1", port), timeout=1)
    client.sendall(
        b"POST /v1/print/raw HTTP/1.0\r\n"
        b"Content-Type: application/octet-stream\r\n"
        b"Content-Length: 5\r\n\r\nabc"
    )
    client.shutdown(socket.SHUT_WR)
    try:
        response = _read_until(client, b"short read")
        assert b"400 Bad Request" in response
        assert b"short read" in response
        assert recorder["writes"] == []
    finally:
        client.close()


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


def test_http_printer_unavailable_is_retryable_service_failure(tmp_path):
    def unavailable(_data):
        raise PrinterUnavailableError("USB receipt printer is unavailable")

    server, svc = make_production_server(
        "127.0.0.1",
        0,
        journal=PrintJournal(str(tmp_path / "jobs.sqlite3")),
        executor=unavailable,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, ctype, data = _post(port, b"hello")
        assert status == 503
        assert ctype.startswith("text/plain")
        assert b"PrinterUnavailableError" in data
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
        assert json.loads(data2) == {"success": True, "bytes": 5}
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
    assert body["device_available"] is True
    assert body["status"] == "ready"
    assert body["last_error"] is None
    assert "queue" in body


def test_livez_reports_service_liveness(http_server):
    port, _ = http_server
    status, ctype, data = _get(port, "/livez")
    assert status == 200
    assert ctype == "application/json"
    body = json.loads(data)
    assert body["live"] is True
    assert body["worker_alive"] is True


def test_device_process_reports_success(monkeypatch):
    sent = []

    class Connection:
        def send(self, value):
            sent.append(value)

        def close(self):
            return None

    monkeypatch.setattr(
        "receipt_print.serve.print_raw_bytes_direct",
        lambda *_args, **_kwargs: None,
    )

    serve_module._device_process(b"receipt", Connection())

    assert sent == [None]


def test_device_process_reports_failure(monkeypatch):
    sent = []

    class Connection:
        def send(self, value):
            sent.append(value)

        def close(self):
            return None

    def fail(*_args, **_kwargs):
        raise OSError("printer unavailable")

    monkeypatch.setattr("receipt_print.serve.print_raw_bytes_direct", fail)

    with pytest.raises(OSError, match="printer unavailable"):
        serve_module._device_process(b"receipt", Connection())

    assert sent == [("OSError", "printer unavailable")]


def test_device_process_reports_system_exit_as_backend_failure(monkeypatch):
    sent = []

    class Connection:
        def send(self, value):
            sent.append(value)

        def close(self):
            return None

    def fail(*_args, **_kwargs):
        raise SystemExit(1)

    monkeypatch.setattr("receipt_print.serve.print_raw_bytes_direct", fail)

    with pytest.raises(SystemExit):
        serve_module._device_process(b"receipt", Connection())

    assert sent == [
        ("RuntimeError", "printer backend exited unexpectedly with status 1")
    ]


def test_healthz_is_not_ready_when_configured_device_is_missing(
    http_server, monkeypatch
):
    port, _ = http_server
    monkeypatch.setenv("RP_DEVICE", "/definitely/missing/receipt-printer")
    monkeypatch.setattr("receipt_print.serve.printer_device_available", lambda: False)
    status, _, data = _get(port, "/healthz")
    body = json.loads(data)
    assert status == 503
    assert body["ready"] is False
    assert body["device_available"] is False

    live_status, _, live_data = _get(port, "/livez")
    live_body = json.loads(live_data)
    assert live_status == 200
    assert live_body["live"] is True
    assert live_body["ready"] is False


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
    assert json.loads(data) == {"success": True, "bytes": len(sample)}
    assert recorder["writes"] == [sample]
    assert len(recorder["writes"]) == 1
    assert recorder["cut_called"] is False


def test_non_contract_raw_path_rejected(http_server):
    port, recorder = http_server
    status, _, _ = _post(port, b"abc", path="/print/raw")
    assert status == 404
    assert recorder["writes"] == []


def test_job_identity_replays_without_duplicate_print(http_server):
    port, recorder = http_server
    first_status, _, first_data = _post(port, b"receipt", job_id="edition:42")
    replay_status, _, replay_data = _post(port, b"receipt", job_id="edition:42")
    assert first_status == 200
    assert replay_status == 200
    assert json.loads(first_data)["replayed"] is False
    assert json.loads(replay_data)["replayed"] is True
    assert recorder["writes"] == [b"receipt"]


def test_job_identity_rejects_different_payload(http_server):
    port, recorder = http_server
    assert _post(port, b"first", job_id="edition:conflict")[0] == 200
    status, ctype, data = _post(port, b"second", job_id="edition:conflict")
    assert status == 409
    assert ctype == "application/json"
    assert json.loads(data)["state"] == "conflict"
    assert recorder["writes"] == [b"first"]


def test_failed_job_identity_can_be_retried(tmp_path):
    writes = []
    attempts = 0

    def fail_once(data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("printer unavailable")
        writes.append(data)

    server, svc = make_production_server(
        "127.0.0.1",
        0,
        journal=PrintJournal(str(tmp_path / "jobs.sqlite3")),
        executor=fail_once,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    retry_port = server.server_address[1]
    try:
        assert _post(retry_port, b"receipt", job_id="edition:retry")[0] == 502
        failed_status, _, failed_data = _get(retry_port, "/healthz")
        assert failed_status == 503
        assert json.loads(failed_data)["status"] == "error"
        status, _, data = _post(retry_port, b"receipt", job_id="edition:retry")
        assert status == 200
        assert json.loads(data)["replayed"] is False
        assert writes == [b"receipt"]
        recovered_status, _, recovered_data = _get(retry_port, "/healthz")
        assert recovered_status == 200
        assert json.loads(recovered_data)["status"] == "ready"
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()


def test_invalid_job_identity_is_rejected(http_server):
    port, recorder = http_server
    status, ctype, _ = _post(port, b"receipt", job_id="not valid")
    assert status == 400
    assert ctype.startswith("text/plain")
    assert recorder["writes"] == []


def test_strip_speed_env_is_structural(monkeypatch):
    monkeypatch.setenv("RP_SPEED_OVERRIDE", "50")
    monkeypatch.setenv("RP_SPEED", "50")
    _strip_speed_env()
    assert os.getenv("RP_SPEED_OVERRIDE") is None
    assert os.getenv("RP_SPEED") is None
    assert os.getenv("RP_HOST") is None
