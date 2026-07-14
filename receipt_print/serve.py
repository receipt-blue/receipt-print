"""Serialized raw-print serving and the shared print executor."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from receipt_print.printer import print_raw_bytes


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


MAX_BYTES = _positive_int_env("RP_SERVE_MAX_BYTES", 8 * 1024 * 1024)
DEFAULT_QUEUE_MAX = _positive_int_env("RP_SERVE_QUEUE_MAX", 32)
DEFAULT_JOB_TIMEOUT = _positive_float_env("RP_SERVE_JOB_TIMEOUT", 30)
DEFAULT_ENQUEUE_TIMEOUT = _positive_float_env("RP_SERVE_ENQUEUE_TIMEOUT", 5)
DEFAULT_DRAIN_GRACE = _positive_float_env("RP_SERVE_DRAIN_GRACE", 10)
DEFAULT_REQUEST_TIMEOUT = _positive_float_env("RP_SERVE_REQUEST_TIMEOUT", 5)

RAW_PATHS = ("/v1/print/raw",)


class PrintQueueFull(Exception):
    """Raised by submit() when the bounded queue cannot accept a job in time."""


class PrintTimeout(Exception):
    """Raised by submit() when a job does not finish within its timeout."""


class PrintServiceStopped(Exception):
    """Raised by submit() after the service has been shut down."""


@dataclass
class _Job:
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    cancelled: bool = False


class PrintService:
    """Run zero-argument printer jobs serially on one worker thread."""

    def __init__(
        self,
        *,
        queue_max: int = DEFAULT_QUEUE_MAX,
        drain_grace: float = DEFAULT_DRAIN_GRACE,
    ) -> None:
        if queue_max < 1:
            raise ValueError("queue_max must be positive")
        if drain_grace <= 0:
            raise ValueError("drain_grace must be positive")
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=queue_max)
        self._drain_grace = drain_grace
        self._worker = threading.Thread(
            target=self._run, name="rp-print-worker", daemon=False
        )
        self._started = False
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._sentinel_enqueued = False
        self._heartbeat = 0.0

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._stopping.is_set():
                raise PrintServiceStopped("print service is stopping")
            self._started = True
        self._worker.start()

    def shutdown(self, *, grace: Optional[float] = None) -> None:
        """Stop accepting work and drain the worker for at most ``grace`` seconds."""
        if not self._started:
            return
        join_grace = grace if grace is not None else self._drain_grace
        if join_grace <= 0:
            raise ValueError("grace must be positive")
        deadline = time.monotonic() + join_grace
        self._stopping.set()
        with self._lock:
            enqueue_sentinel = not self._sentinel_enqueued
            self._sentinel_enqueued = True
        if enqueue_sentinel:
            self._enqueue_sentinel(deadline)
        if threading.current_thread() is not self._worker:
            remaining = max(0.0, deadline - time.monotonic())
            self._worker.join(timeout=remaining)

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float = DEFAULT_JOB_TIMEOUT,
        enqueue_timeout: float = DEFAULT_ENQUEUE_TIMEOUT,
    ) -> Any:
        """Enqueue a job, wait for its result, and propagate its exception."""
        if self._stopping.is_set() or not self._started:
            raise PrintServiceStopped("print service is not accepting jobs")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if enqueue_timeout <= 0:
            raise ValueError("enqueue_timeout must be positive")
        job = _Job(fn=fn)
        deadline = time.monotonic() + enqueue_timeout
        while True:
            if self._stopping.is_set():
                raise PrintServiceStopped("print service is not accepting jobs")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PrintQueueFull("print queue is full")
            try:
                self._queue.put(job, timeout=min(remaining, 0.1))
                break
            except queue.Full:
                continue
        if not job.done.wait(timeout=timeout):
            with job.state_lock:
                if not job.started:
                    job.cancelled = True
            raise PrintTimeout(f"print job did not complete within {timeout}s")
        if job.error is not None:
            raise job.error
        return job.result

    def health(self) -> dict[str, Any]:
        worker_alive = self._worker.is_alive()
        ready = worker_alive and not self._stopping.is_set()
        return {
            "ok": ready,
            "ready": ready,
            "worker_alive": worker_alive,
            "queue": self._queue.qsize(),
            "heartbeat_age": (time.monotonic() - self._heartbeat)
            if self._heartbeat
            else None,
            "stopping": self._stopping.is_set(),
        }

    def _cancel_queued_jobs(self) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not None:
                with job.state_lock:
                    if not job.started:
                        job.cancelled = True
                        job.error = PrintServiceStopped("print service stopped before the job ran")
                        job.done.set()
            self._queue.task_done()

    def _enqueue_sentinel(self, deadline: float) -> None:
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    self._cancel_queued_jobs()
                    try:
                        self._queue.put_nowait(None)
                        return
                    except queue.Full:
                        continue
                time.sleep(min(0.01, deadline - time.monotonic()))

    def _run(self) -> None:
        while True:
            self._heartbeat = time.monotonic()
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            with job.state_lock:
                if job.cancelled:
                    self._queue.task_done()
                    continue
                job.started = True
            try:
                job.result = job.fn()
            except BaseException as exc:
                job.error = exc
            finally:
                job.done.set()
                self._queue.task_done()


def make_raw_handler(
    service: PrintService,
    *,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> type[BaseHTTPRequestHandler]:
    """Build the HTTP handler for the kiosk-compatible raw endpoint."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if job_timeout <= 0:
        raise ValueError("job_timeout must be positive")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    class RawHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(request_timeout)

        def _text(self, status: int, message: str) -> None:
            body = message.encode("utf-8", "replace")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_error(
            self,
            code: int,
            message: Optional[str] = None,
            explain: Optional[str] = None,
        ) -> None:
            del explain
            default_message = self.responses.get(code, ("error",))[0]
            self.close_connection = True
            self._text(code, message or default_message)

        def do_GET(self) -> None:
            self.close_connection = True
            if self.path == "/healthz":
                health = service.health()
                self._json(200 if health["ready"] else 503, health)
                return
            self._text(404, "not found")

        def do_POST(self) -> None:
            self.close_connection = True
            if self.path not in RAW_PATHS:
                self._text(404, "not found")
                return
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/octet-stream":
                self._text(415, "content type must be application/octet-stream")
                return
            if self.headers.get("Transfer-Encoding"):
                self._text(400, "transfer encoding is not supported")
                return
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) != 1:
                self._text(400, "Content-Length is required")
                return
            raw_length = lengths[0].strip()
            if not re.fullmatch(r"[0-9]+", raw_length, flags=re.ASCII):
                self._text(400, "invalid Content-Length")
                return
            normalized_length = raw_length.lstrip("0") or "0"
            max_length = str(max_bytes)
            if len(normalized_length) > len(max_length) or (
                len(normalized_length) == len(max_length)
                and normalized_length > max_length
            ):
                self._text(413, f"body exceeds {max_bytes} bytes")
                return
            length = int(normalized_length)
            if length == 0:
                self._text(400, "empty body")
                return
            try:
                body = self.rfile.read(length)
            except TimeoutError:
                self._text(408, "request body timed out")
                return
            except OSError:
                self._text(400, "body read failed")
                return
            if len(body) != length:
                self._text(400, "short read")
                return
            try:
                service.submit(
                    lambda: print_raw_bytes(body, cut=False), timeout=job_timeout
                )
            except PrintQueueFull as exc:
                self._text(503, f"{type(exc).__name__}: {exc}")
                return
            except PrintServiceStopped as exc:
                self._text(503, f"{type(exc).__name__}: {exc}")
                return
            except PrintTimeout as exc:
                self._text(504, f"{type(exc).__name__}: {exc}")
                return
            except BaseException as exc:
                self._text(502, f"{type(exc).__name__}: {exc}")
                return
            self._json(200, {"success": True, "bytes": len(body)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return RawHandler


def make_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    service: Optional[PrintService] = None,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> tuple[ThreadingHTTPServer, PrintService]:
    """Build (but do not serve) a standalone raw-print server bound to host:port.

    Returns (server, service) so tests can bind 127.0.0.1:0 and tear down.
    """
    svc = service or PrintService()
    handler = make_raw_handler(
        svc,
        max_bytes=max_bytes,
        job_timeout=job_timeout,
        request_timeout=request_timeout,
    )
    server = _PrintHTTPServer((host, port), handler)
    try:
        svc.start()
    except BaseException:
        server.server_close()
        raise
    return server, svc


def _strip_speed_env() -> None:
    os.environ.pop("RP_SPEED_OVERRIDE", None)
    os.environ.pop("RP_SPEED", None)


def run_server(
    host: str = "127.0.0.1",
    port: int = 9100,
    *,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> None:
    """Run the standalone server for non-kiosk clients."""
    _strip_speed_env()
    server, svc = make_server(
        host,
        port,
        max_bytes=max_bytes,
        job_timeout=job_timeout,
        request_timeout=request_timeout,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        svc.shutdown()


class _PrintHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
