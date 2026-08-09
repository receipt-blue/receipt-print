import base64
import json
import subprocess
import sys

import pytest

from receipt_print.receipt_core import (
    CORE_PROTOCOL_VERSION,
    LocalReceiptCore,
    ReceiptCoreClient,
    ReceiptCoreError,
)


class Response:
    def __init__(self, payload, *, ok=True, status_code=200):
        self.payload = payload
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return self.response


def completed(payload, returncode=0, stderr=b""):
    return subprocess.CompletedProcess(
        ["receipt-core"],
        returncode,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=stderr,
    )


def test_default_backend_is_local_and_ignores_substrate_url(monkeypatch):
    monkeypatch.delenv("RECEIPT_CORE_URL", raising=False)
    monkeypatch.setenv("RECEIPT_CORE_BIN", sys.executable)
    monkeypatch.setenv("SUBSTRATE_URL", "http://127.0.0.1:3082")

    client = ReceiptCoreClient()

    assert client.mode == "local"
    assert client.base_url is None


def test_explicit_core_url_selects_remote_substrate(monkeypatch):
    monkeypatch.setenv("RECEIPT_CORE_URL", "http://127.0.0.1:3081/")

    client = ReceiptCoreClient(session=Session(Response({})))

    assert client.mode == "remote"
    assert client.base_url == "http://127.0.0.1:3081"


def test_local_submit_uses_versioned_ipc_and_decodes_bytes():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed(
            {
                "v": CORE_PROTOCOL_VERSION,
                "rendererVersion": "renderer/1",
                "profile": "TM-T20II",
                "bytes": base64.b64encode(b"escpos").decode("ascii"),
                "economy": {"dotLines": 24},
            }
        )

    core = LocalReceiptCore(sys.executable, runner=runner)
    result = core.submit({"format": "document/1"})

    request = json.loads(calls[0][1]["input"])
    assert calls[0][0] == [sys.executable, "render"]
    assert request == {
        "v": CORE_PROTOCOL_VERSION,
        "operation": "render",
        "document": {"format": "document/1"},
        "profile": "TM-T20II",
    }
    assert result.bytes == b"escpos"
    assert result.edition_id is None
    assert result.renderer_version == "renderer/1"


def test_local_failure_surfaces_stderr_without_an_http_fallback():
    def runner(argv, **kwargs):
        return completed({}, returncode=1, stderr=b"receipt-core: invalid document\n")

    core = LocalReceiptCore(sys.executable, runner=runner)

    with pytest.raises(ReceiptCoreError, match="invalid document"):
        core.submit({"format": "document/1"})


def test_remote_preview_sends_study_and_reads_evaluation_fields():
    session = Session(
        Response(
            {
                "v": "wire/1",
                "png": "cG5n",
                "pngs": ["cG5n", "cG5nMg=="],
                "text": "preview",
                "economy": {"dotLines": 10},
                "regions": [{"path": "/blocks/0", "page": 0}],
                "qrResults": [
                    {"payload": "https://example.org", "decoded": True}
                ],
            }
        )
    )
    client = ReceiptCoreClient("http://core.test", session=session)

    result = client.preview(
        {"format": "document/1"},
        study={"orientation": "landscape", "widthDots": 960},
    )

    assert session.calls == [
        (
            "http://core.test/v1/preview",
            {
                "v": "wire/1",
                "document": {"format": "document/1"},
                "study": {"orientation": "landscape", "widthDots": 960},
            },
            90,
        )
    ]
    assert result.pngs == ("cG5n", "cG5nMg==")
    assert result.regions[0]["path"] == "/blocks/0"
    assert result.qr_results[0]["decoded"] is True


def test_remote_error_uses_wire_message():
    client = ReceiptCoreClient(
        "http://core.test",
        session=Session(
            Response(
                {"error": {"message": "invalid qr rail"}},
                ok=False,
                status_code=400,
            )
        ),
    )

    with pytest.raises(
        ReceiptCoreError,
        match=r"invalid qr rail \[http://core\.test\]",
    ):
        client.preview({"format": "document/1"})
