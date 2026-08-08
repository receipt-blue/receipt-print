from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import requests


class ReceiptCoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreviewResult:
    png: str
    pngs: tuple[str, ...]
    text: str
    economy: dict[str, Any]
    regions: tuple[dict[str, Any], ...]
    qr_results: tuple[dict[str, Any], ...]


class ReceiptCoreClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: int = 90,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("RECEIPT_CORE_URL")
            or "http://127.0.0.1:3080"
        ).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}{path}", json=body, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ReceiptCoreError(
                f"Could not reach receipt-core at {self.base_url}: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ReceiptCoreError(
                f"receipt-core returned HTTP {response.status_code} without JSON"
            ) from exc
        if not response.ok:
            error = payload.get("error") if isinstance(payload, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            raise ReceiptCoreError(
                f"{message or f'receipt-core returned HTTP {response.status_code}'} "
                f"[{self.base_url}]"
            )
        return payload

    def preview(
        self,
        document: dict[str, Any],
        *,
        study: Optional[dict[str, Any]] = None,
    ) -> PreviewResult:
        body: dict[str, Any] = {"v": "wire/1", "document": document}
        if study:
            body["study"] = study
        payload = self._post("/v1/preview", body)
        pngs = payload.get("pngs") or [payload["png"]]
        return PreviewResult(
            png=payload["png"],
            pngs=tuple(pngs),
            text=payload.get("text", ""),
            economy=payload.get("economy", {}),
            regions=tuple(payload.get("regions", [])),
            qr_results=tuple(payload.get("qrResults", [])),
        )

    def print_document(self, document: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/print", {"v": "wire/1", "document": document})
