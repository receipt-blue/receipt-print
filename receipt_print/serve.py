"""Shape-agnostic serialized print serving for receipt-print.

`PrintService` runs OPAQUE printer jobs (zero-arg callables) one at a time on a
single non-daemon worker thread; it never inspects or assumes what a job prints.
The raw-bytes HTTP handler factory and the standalone ThreadingHTTPServer wrapper
are the receipt-print-specific clients of that general executor; the kiosk reuses
`PrintService` directly to serialize its own heterogeneous print paths.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from receipt_print.printer import print_raw_bytes

MAX_BYTES = int(os.getenv("RP_SERVE_MAX_BYTES", str(8 * 1024 * 1024)))
DEFAULT_QUEUE_MAX = int(os.getenv("RP_SERVE_QUEUE_MAX", "32"))
DEFAULT_JOB_TIMEOUT = float(os.getenv("RP_SERVE_JOB_TIMEOUT", "30"))
DEFAULT_ENQUEUE_TIMEOUT = float(os.getenv("RP_SERVE_ENQUEUE_TIMEOUT", "5"))
DEFAULT_DRAIN_GRACE = float(os.getenv("RP_SERVE_DRAIN_GRACE", "10"))

RAW_PATHS = ("/v1/print/raw", "/print/raw")


class PrintQueueFull(Exception):
    """Raised by submit() when the bounded queue cannot accept a job in time."""


class PrintTimeout(Exception):
    """Raised by submit() when a job does not finish within its timeout (job is abandoned)."""


class PrintServiceStopped(Exception):
    """Raised by submit() after the service has been shut down."""


@dataclass
class _Job:
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    abandoned: bool = False


class PrintService:
    """Serialize OPAQUE printer jobs onto one worker thread.

    A job is any zero-arg callable. PrintService runs it, captures its return value
    or BaseException, and never assumes what it prints. Connect-per-job lives inside
    the job (e.g. print_raw_bytes does open->_raw->close). The single worker IS the
    serialization (python-escpos is not thread-safe).
    """

    def __init__(
        self,
        *,
        queue_max: int = DEFAULT_QUEUE_MAX,
        drain_grace: float = DEFAULT_DRAIN_GRACE,
    ) -> None:
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=queue_max)
        self._drain_grace = drain_grace
        self._worker = threading.Thread(
            target=self._run, name="rp-print-worker", daemon=False
        )
        self._started = False
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._heartbeat = 0.0
        self._alive = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._worker.start()

    def shutdown(self, *, grace: Optional[float] = None) -> None:
        """Sentinel + join within grace so an in-flight job finishes; idempotent."""
        if not self._started:
            return
        join_grace = grace if grace is not None else self._drain_grace
        if self._stopping.is_set():
            self._worker.join(timeout=join_grace)
            return
        self._stopping.set()
        self._queue.put(None)
        self._worker.join(timeout=join_grace)

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float = DEFAULT_JOB_TIMEOUT,
        enqueue_timeout: float = DEFAULT_ENQUEUE_TIMEOUT,
    ) -> Any:
        """Enqueue an opaque job, block until it finishes, return fn()'s value.

        Raises PrintServiceStopped if shut down, PrintQueueFull if the bounded queue
        stays full past enqueue_timeout, PrintTimeout (and ABANDONS the job so it will
        NOT print later) if it does not finish in `timeout`, or re-raises the job's own
        BaseException (incl. SystemExit from connect_printer's sys.exit).
        """
        if self._stopping.is_set() or not self._started:
            raise PrintServiceStopped("print service is not accepting jobs")
        job = _Job(fn=fn)
        try:
            self._queue.put(job, timeout=enqueue_timeout)
        except queue.Full as exc:
            raise PrintQueueFull("print queue is full") from exc
        if not job.done.wait(timeout=timeout):
            job.abandoned = True
            raise PrintTimeout(f"print job did not complete within {timeout}s")
        if job.error is not None:
            raise job.error
        return job.result

    def health(self) -> dict[str, Any]:
        return {
            "worker_alive": self._worker.is_alive(),
            "queue": self._queue.qsize(),
            "heartbeat_age": (time.monotonic() - self._heartbeat)
            if self._heartbeat
            else None,
            "stopping": self._stopping.is_set(),
        }

    def _run(self) -> None:
        self._alive = True
        try:
            while True:
                self._heartbeat = time.monotonic()
                job = self._queue.get()
                if job is None:
                    self._queue.task_done()
                    return
                if job.abandoned:
                    self._queue.task_done()
                    continue
                try:
                    job.result = job.fn()
                except BaseException as exc:
                    job.error = exc
                finally:
                    job.done.set()
                    self._queue.task_done()
        finally:
            self._alive = False


def make_raw_handler(
    service: PrintService,
    *,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
) -> type[BaseHTTPRequestHandler]:
    """HTTP handler mapping a POSTed octet-stream body to a print_raw_bytes job.

    Success => JSON {"ok": true, "bytes": n}. EVERY error => text/plain so that
    receipt-wiki's receiptResponse surfaces the reason instead of an opaque status.
    """

    class RawHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

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

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/healthz":
                self._json(200, service.health())
                return
            self._text(404, "not found")

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] not in RAW_PATHS:
                self._text(404, "not found")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > max_bytes:
                self._text(413, f"body exceeds {max_bytes} bytes")
                return
            body = self.rfile.read(length) if length > 0 else b""
            if len(body) != length:
                self._text(400, "short read")
                return
            if not body:
                self._text(400, "empty body")
                return
            try:
                service.submit(
                    lambda: print_raw_bytes(body, cut=False), timeout=job_timeout
                )
            except BaseException as exc:
                self._text(502, f"{type(exc).__name__}: {exc}")
                return
            self._json(200, {"ok": True, "bytes": len(body)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return RawHandler


def make_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    service: Optional[PrintService] = None,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
) -> tuple[ThreadingHTTPServer, PrintService]:
    """Build (but do not serve) a standalone raw-print server bound to host:port.

    Returns (server, service) so tests can bind 127.0.0.1:0 and tear down.
    """
    svc = service or PrintService()
    svc.start()
    handler = make_raw_handler(svc, max_bytes=max_bytes, job_timeout=job_timeout)
    server = ThreadingHTTPServer((host, port), handler)
    return server, svc


def _strip_speed_env() -> None:
    """Structural guardrail 5: --speed must never inject GS ( K on the serve wire."""
    os.environ.pop("RP_SPEED_OVERRIDE", None)
    os.environ.pop("RP_SPEED", None)


def run_server(
    host: str = "127.0.0.1",
    port: int = 9100,
    *,
    max_bytes: int = MAX_BYTES,
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
) -> None:
    """Standalone serve loop for NON-kiosk clients (Godot/API/dev).

    Pops RP_SPEED_OVERRIDE/RP_SPEED structurally so --speed cannot inject GS ( K.
    Leaves RP_HOST untouched so connect_printer() uses USB rather than silently
    routing to a Network(host=...) that would hang.
    """
    _strip_speed_env()
    server, svc = make_server(host, port, max_bytes=max_bytes, job_timeout=job_timeout)
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()
        svc.shutdown()
