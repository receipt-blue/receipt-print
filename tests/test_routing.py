import json

import pytest
import requests

from receipt_print.routing import (
    DeviceLock,
    ServicePrinter,
    print_mode,
    service_ready,
    submit_raw,
)


class Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.reason = text

    def json(self):
        return self._payload


def test_print_mode_defaults_to_direct(monkeypatch):
    monkeypatch.delenv("RP_PRINT_MODE", raising=False)
    assert print_mode() == "direct"


def test_print_mode_rejects_implicit_auto_routing(monkeypatch):
    monkeypatch.setenv("RP_PRINT_MODE", "auto")
    with pytest.raises(RuntimeError, match="service or direct"):
        print_mode()


def test_service_ready_requires_ready_payload(monkeypatch):
    monkeypatch.setattr(
        "receipt_print.routing.requests.get",
        lambda *args, **kwargs: Response(payload={"ready": True}),
    )
    assert service_ready("http://printer") is True

    monkeypatch.setattr(
        "receipt_print.routing.requests.get",
        lambda *args, **kwargs: Response(payload={"ready": False}),
    )
    assert service_ready("http://printer") is False


def test_service_ready_treats_connection_failure_as_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("receipt_print.routing.requests.get", unavailable)
    assert service_ready("http://printer") is False


def test_submit_raw_sends_stable_identity(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response(
            payload={
                "success": True,
                "job_id": "edition:42",
                "state": "printed",
            }
        )

    monkeypatch.setattr("receipt_print.routing.requests.post", post)
    result = submit_raw(
        b"receipt",
        job_id="edition:42",
        url="http://printer",
        title="FIFA World Cup™",
        source="receipt wiki",
    )
    assert result["state"] == "printed"
    assert seen["url"] == "http://printer/v1/print/raw"
    assert seen["data"] == b"receipt"
    assert seen["headers"]["X-Receipt-Print-Job-Id"] == "edition:42"
    assert seen["headers"]["X-Receipt-Print-Title"] == "FIFA%20World%20Cup%E2%84%A2"
    assert seen["headers"]["X-Receipt-Print-Source"] == "receipt%20wiki"


def test_submit_raw_reports_ambiguous_connection_failure(monkeypatch):
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(kwargs["headers"]["X-Receipt-Print-Job-Id"])
        raise requests.ConnectionError("response lost")

    monkeypatch.setattr("receipt_print.routing.requests.post", unavailable)
    with pytest.raises(RuntimeError, match="may have printed"):
        submit_raw(b"receipt", job_id="edition:42", url="http://printer")
    assert attempts == ["edition:42", "edition:42"]


def test_submit_raw_recovers_lost_response_without_changing_identity(monkeypatch):
    attempts = []

    def post(*args, **kwargs):
        identity = kwargs["headers"]["X-Receipt-Print-Job-Id"]
        attempts.append(identity)
        if len(attempts) == 1:
            raise requests.ConnectionError("response lost")
        return Response(
            payload={
                "success": True,
                "job_id": identity,
                "state": "printed",
                "replayed": True,
            }
        )

    monkeypatch.setattr("receipt_print.routing.requests.post", post)
    result = submit_raw(b"receipt", job_id="edition:42", url="http://printer")
    assert result["state"] == "printed"
    assert result["replayed"] is True
    assert attempts == ["edition:42", "edition:42"]


def test_service_printer_submits_rendered_bytes(monkeypatch):
    seen = {}

    def submit(data, **kwargs):
        seen["data"] = data
        seen.update(kwargs)
        return {"success": True}

    monkeypatch.setattr("receipt_print.routing.submit_raw", submit)
    printer = ServicePrinter(
        profile="TM-T20II",
        charcode="CP437",
        speed=None,
        apply_speed=lambda *args: None,
        url="http://printer",
    )
    printer.text("hello")
    expected = printer.output
    printer.close()
    printer.close()
    assert seen == {"data": expected, "url": "http://printer"}


def test_service_printer_flushes_complete_batches(monkeypatch):
    submitted = []
    monkeypatch.setenv("RP_SERVICE_BATCH_BYTES", "1")
    monkeypatch.setattr(
        "receipt_print.routing.submit_raw",
        lambda data, **kwargs: submitted.append((data, kwargs)),
    )
    printer = ServicePrinter(
        profile="TM-T20II",
        charcode="CP437",
        speed=None,
        apply_speed=lambda *args: None,
        url="http://printer",
    )
    printer.text("first")
    first = printer.output
    printer.flush_pending()
    assert printer.output == b""

    printer.text("second")
    second = printer.output
    printer.close()

    assert submitted == [
        (first, {"url": "http://printer"}),
        (second, {"url": "http://printer"}),
    ]


def test_service_printer_does_not_resubmit_failed_batch_on_close(monkeypatch):
    attempts = []
    monkeypatch.setenv("RP_SERVICE_BATCH_BYTES", "1")

    def fail(data, **kwargs):
        attempts.append((data, kwargs))
        raise RuntimeError("pipe error")

    monkeypatch.setattr("receipt_print.routing.submit_raw", fail)
    printer = ServicePrinter(
        profile="TM-T20II",
        charcode="CP437",
        speed=None,
        apply_speed=lambda *args: None,
        url="http://printer",
    )
    printer.text("payload")

    with pytest.raises(RuntimeError, match="pipe error"):
        printer.flush_pending()
    printer.close()

    assert len(attempts) == 1


def test_device_lock_serializes_and_times_out(tmp_path):
    path = tmp_path / "printer.lock"
    first = DeviceLock(str(path), timeout=1)
    second = DeviceLock(str(path), timeout=0.05)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="remained busy"):
            second.acquire()
    finally:
        first.release()


def test_journal_payload_is_json_serializable(monkeypatch):
    monkeypatch.setattr(
        "receipt_print.routing.requests.post",
        lambda *args, **kwargs: Response(
            payload={"success": True, "job_id": "job-1", "state": "printed"}
        ),
    )
    assert json.dumps(submit_raw(b"x", job_id="job-1"))


@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "job_id": "job-1"},
        {"success": True, "job_id": "job-1", "state": "ambiguous"},
        {"success": True, "job_id": "different", "state": "printed"},
        {"success": False, "job_id": "job-1", "state": "printed"},
    ],
)
def test_submit_raw_requires_exact_printed_confirmation(monkeypatch, payload):
    monkeypatch.setattr(
        "receipt_print.routing.requests.post",
        lambda *args, **kwargs: Response(payload=payload),
    )
    with pytest.raises(RuntimeError, match="truthfully confirm"):
        submit_raw(b"x", job_id="job-1", url="http://printer")
