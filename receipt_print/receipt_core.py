from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Protocol

import requests


CORE_PROTOCOL_VERSION = "receipt-core/render/1"


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
    renderer_version: Optional[str] = None
    profile: Optional[str] = None
    fidelity: Optional[dict[str, Any]] = None
    measurement: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class SubmissionResult:
    bytes: Optional[bytes]
    edition_id: Optional[str]
    economy: dict[str, Any]
    renderer_version: Optional[str] = None
    profile: Optional[str] = None


class ReceiptCoreBackend(Protocol):
    mode: str

    def close(self) -> None: ...

    def preview(
        self,
        document: dict[str, Any],
        *,
        study: Optional[dict[str, Any]] = None,
    ) -> PreviewResult: ...

    def submit(self, document: dict[str, Any]) -> SubmissionResult: ...


def _companion_binary() -> Optional[str]:
    try:
        companion = import_module("receipt_core_renderer")
    except ImportError:
        return None
    renderer_path = getattr(companion, "renderer_path", None)
    if not callable(renderer_path):
        raise ReceiptCoreError(
            "receipt-core-renderer is installed but does not expose renderer_path()"
        )
    return os.fspath(renderer_path())


def resolve_core_binary(explicit: Optional[str] = None) -> str:
    candidates = [
        explicit,
        os.getenv("RECEIPT_CORE_BIN"),
        _companion_binary(),
        shutil.which("receipt-core"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise ReceiptCoreError(
        "receipt-core executable is unavailable; install receipt-core-renderer, "
        "use the Nix package, or set RECEIPT_CORE_BIN"
    )


class LocalReceiptCore:
    mode = "local"

    def __init__(
        self,
        executable: Optional[str] = None,
        *,
        timeout: int = 90,
        profile: Optional[str] = None,
        runner=subprocess.run,
    ):
        self.executable = resolve_core_binary(executable)
        self.timeout = timeout
        self.profile = profile or os.getenv("RP_PROFILE") or "TM-T20II"
        self._runner = runner

    def close(self) -> None:
        return None

    def _invoke(
        self,
        operation: str,
        document: dict[str, Any],
        *,
        study: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "v": CORE_PROTOCOL_VERSION,
            "operation": operation,
            "document": document,
            "profile": self.profile,
        }
        if study is not None:
            request["study"] = study
        try:
            completed = self._runner(
                [self.executable, operation],
                input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReceiptCoreError(f"could not invoke receipt-core: {exc}") from exc
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            detail = stderr or f"receipt-core exited {completed.returncode}"
            raise ReceiptCoreError(detail)
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptCoreError(
                "receipt-core returned an invalid JSON response"
            ) from exc
        if not isinstance(payload, dict) or payload.get("v") != CORE_PROTOCOL_VERSION:
            raise ReceiptCoreError(
                f"receipt-core returned an incompatible response; expected {CORE_PROTOCOL_VERSION}"
            )
        return payload

    def preview(
        self,
        document: dict[str, Any],
        *,
        study: Optional[dict[str, Any]] = None,
    ) -> PreviewResult:
        payload = self._invoke("preview", document, study=study)
        png = payload.get("png")
        if not isinstance(png, str):
            raise ReceiptCoreError("receipt-core preview response is missing png")
        pngs = payload.get("pngs") or [png]
        return PreviewResult(
            png=png,
            pngs=tuple(pngs),
            text=payload.get("text", ""),
            economy=payload.get("economy", {}),
            regions=tuple(payload.get("regions", [])),
            qr_results=tuple(payload.get("qrResults", [])),
            renderer_version=payload.get("rendererVersion"),
            profile=payload.get("profile"),
            fidelity=payload.get("fidelity"),
            measurement=payload.get("measurement"),
        )

    def submit(self, document: dict[str, Any]) -> SubmissionResult:
        payload = self._invoke("render", document)
        encoded = payload.get("bytes")
        if not isinstance(encoded, str):
            raise ReceiptCoreError("receipt-core render response is missing bytes")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ReceiptCoreError(
                "receipt-core render response contains invalid base64 bytes"
            ) from exc
        return SubmissionResult(
            bytes=data,
            edition_id=None,
            economy=payload.get("economy", {}),
            renderer_version=payload.get("rendererVersion"),
            profile=payload.get("profile"),
        )


class RemoteReceiptSubstrate:
    mode = "remote"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int = 90,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
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
                f"could not reach receipt-substrate at {self.base_url}: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ReceiptCoreError(
                f"receipt-substrate returned HTTP {response.status_code} without JSON"
            ) from exc
        if not response.ok:
            error = payload.get("error") if isinstance(payload, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            raise ReceiptCoreError(
                f"{message or f'receipt-substrate returned HTTP {response.status_code}'} "
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
        if study is not None:
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

    def submit(self, document: dict[str, Any]) -> SubmissionResult:
        payload = self._post("/v1/print", {"v": "wire/1", "document": document})
        return SubmissionResult(
            bytes=None,
            edition_id=payload.get("editionId"),
            economy=payload.get("economy", {}),
        )


class ReceiptCoreClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        executable: Optional[str] = None,
        timeout: int = 90,
        session: Optional[requests.Session] = None,
        runner=subprocess.run,
    ):
        remote_url = base_url or os.getenv("RECEIPT_CORE_URL")
        if remote_url is not None or session is not None:
            self.backend: ReceiptCoreBackend = RemoteReceiptSubstrate(
                remote_url or "http://127.0.0.1:3080",
                timeout=timeout,
                session=session,
            )
        else:
            self.backend = LocalReceiptCore(
                executable,
                timeout=timeout,
                runner=runner,
            )

    @property
    def mode(self) -> str:
        return self.backend.mode

    @property
    def base_url(self) -> Optional[str]:
        return getattr(self.backend, "base_url", None)

    def close(self) -> None:
        self.backend.close()

    def preview(
        self,
        document: dict[str, Any],
        *,
        study: Optional[dict[str, Any]] = None,
    ) -> PreviewResult:
        return self.backend.preview(document, study=study)

    def submit(self, document: dict[str, Any]) -> SubmissionResult:
        return self.backend.submit(document)
