import json

import pytest
import requests

from receipt_print.routing import (
    DeviceLock,
    ServicePrinter,
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
    result = submit_raw(b"receipt", job_id="edition:42", url="http://printer")
    assert result["state"] == "printed"
    assert seen["url"] == "http://printer/v1/print/raw"
    assert seen["data"] == b"receipt"
    assert seen["headers"]["X-Receipt-Print-Job-Id"] == "edition:42"


def test_submit_raw_reports_ambiguous_connection_failure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("response lost")

    monkeypatch.setattr("receipt_print.routing.requests.post", unavailable)
    with pytest.raises(RuntimeError, match="may have printed"):
        submit_raw(b"receipt", job_id="edition:42", url="http://printer")


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
