import pytest

from receipt_print.receipt_core import ReceiptCoreClient, ReceiptCoreError


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


def test_core_url_defaults_to_local_receipt_core(monkeypatch):
    monkeypatch.delenv("RECEIPT_CORE_URL", raising=False)
    monkeypatch.setenv("SUBSTRATE_URL", "http://127.0.0.1:3082")

    client = ReceiptCoreClient(session=Session(Response({})))

    assert client.base_url == "http://127.0.0.1:3080"


def test_core_url_honors_receipt_core_specific_environment(monkeypatch):
    monkeypatch.setenv("RECEIPT_CORE_URL", "http://127.0.0.1:3081/")

    client = ReceiptCoreClient(session=Session(Response({})))

    assert client.base_url == "http://127.0.0.1:3081"


def test_preview_sends_study_and_reads_additive_evaluation_fields():
    session = Session(
        Response(
            {
                "v": "wire/1",
                "png": "cG5n",
                "pngs": ["cG5n", "cG5nMg=="],
                "text": "preview",
                "economy": {"dotLines": 10},
                "regions": [{"path": "/blocks/0", "page": 0}],
                "qrResults": [{"payload": "https://example.org", "decoded": True}],
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


def test_core_error_uses_wire_message():
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
