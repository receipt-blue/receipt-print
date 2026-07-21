from __future__ import annotations

import fcntl
import os
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

import requests
from escpos.printer import Dummy


DEFAULT_SERVICE_URL = "http://127.0.0.1:9100"
DEFAULT_HEALTH_TIMEOUT = 0.25
DEFAULT_LOCK_TIMEOUT = 35.0
DEFAULT_LOCK_PATH = "/tmp/receipt-print-device.lock"


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


def _optional_positive_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, "", "0"):
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be zero or a positive number") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be zero or a positive number")
    return value


def print_mode() -> str:
    mode = os.getenv("RP_PRINT_MODE", "auto").strip().lower()
    if mode not in {"auto", "service", "direct"}:
        raise RuntimeError("RP_PRINT_MODE must be auto, service, or direct")
    return mode


def service_url() -> str:
    return os.getenv("RP_SERVICE_URL", DEFAULT_SERVICE_URL).rstrip("/")


def service_ready(url: str | None = None) -> bool:
    target = url if url is not None else service_url()
    if not target:
        return False
    timeout = _positive_float_env("RP_SERVICE_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT)
    try:
        response = requests.get(f"{target}/healthz", timeout=timeout)
        if response.status_code != 200:
            return False
        payload = response.json()
        return payload.get("ready") is True
    except (requests.RequestException, ValueError):
        return False


def submit_raw(
    data: bytes,
    *,
    job_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    source: str | None = None,
) -> dict:
    target = url if url is not None else service_url()
    if not target:
        raise RuntimeError("receipt-print service URL is disabled")
    timeout = _optional_positive_float_env("RP_SERVICE_DELIVERY_TIMEOUT")
    identity = job_id or str(uuid.uuid4())
    response = None
    last_error = None
    for _attempt in range(2):
        try:
            headers = {
                "Content-Type": "application/octet-stream",
                "X-Receipt-Print-Job-Id": identity,
            }
            if title:
                headers["X-Receipt-Print-Title"] = urllib.parse.quote(title)
            if source:
                headers["X-Receipt-Print-Source"] = urllib.parse.quote(source)
            response = requests.post(
                f"{target}/v1/print/raw",
                data=data,
                headers=headers,
                timeout=(5, timeout),
            )
            break
        except requests.RequestException as exc:
            last_error = exc
    if response is None:
        raise RuntimeError(
            f"receipt-print service delivery failed; job {identity} may have printed: {last_error}"
        ) from last_error
    if response.status_code != 200:
        message = response.text.strip() or response.reason
        raise RuntimeError(
            f"receipt-print service rejected job {identity} "
            f"with HTTP {response.status_code}: {message}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"receipt-print service returned an invalid response for job {identity}"
        ) from exc
    if (
        payload.get("success") is not True
        or payload.get("state") != "printed"
        or payload.get("job_id") != identity
    ):
        raise RuntimeError(
            f"receipt-print service did not truthfully confirm job {identity} as printed"
        )
    return payload


class DeviceLock:
    def __init__(self, path: str | None = None, timeout: float | None = None) -> None:
        self.path = Path(
            path or os.getenv("RP_DEVICE_LOCK_PATH", DEFAULT_LOCK_PATH)
        )
        self.timeout = timeout or _positive_float_env(
            "RP_DEVICE_LOCK_TIMEOUT", DEFAULT_LOCK_TIMEOUT
        )
        self._handle = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.release()
                    raise RuntimeError(
                        f"printer device remained busy for {self.timeout:g}s"
                    )
                time.sleep(min(0.05, deadline - time.monotonic()))

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


class LockedPrinter:
    def __init__(self, printer, lock: DeviceLock) -> None:
        self._printer = printer
        self._lock = lock
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._printer, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._printer, "close", None)
            if callable(close):
                close()
        finally:
            self._lock.release()


class ServicePrinter(Dummy):
    def __init__(
        self,
        *,
        profile: str,
        charcode: str,
        speed: int | None,
        apply_speed: Callable,
        url: str,
    ) -> None:
        super().__init__(profile=profile)
        self.charcode(charcode)
        apply_speed(self, speed)
        self._service_url = url
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        data = self.output
        super().close()
        if data:
            submit_raw(data, url=self._service_url)
