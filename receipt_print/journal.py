from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


JOB_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z", flags=re.ASCII)


@dataclass(frozen=True)
class JobClaim:
    job_id: str
    digest: str
    state: str
    replayed: bool


class PrintJournal:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._recover_inflight()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    error TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "title" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN title TEXT")
            if "source" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN source TEXT")

    def _recover_inflight(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET state = 'ambiguous',
                    error = 'print service restarted before delivery was confirmed',
                    updated_at = ?
                WHERE state = 'printing'
                """,
                (time.time(),),
            )

    def claim(
        self,
        job_id: str,
        data: bytes,
        *,
        title: str | None = None,
        source: str | None = None,
    ) -> JobClaim:
        if not JOB_ID_RE.fullmatch(job_id):
            raise ValueError(
                "X-Receipt-Print-Job-Id must contain 1-128 letters, digits, ., _, :, or -"
            )
        digest = hashlib.sha256(data).hexdigest()
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT digest, state FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, digest, state, byte_count, error, updated_at, title, source
                    )
                    VALUES (?, ?, 'printing', ?, NULL, ?, ?, ?)
                    """,
                    (job_id, digest, len(data), now, title, source),
                )
                connection.commit()
                return JobClaim(job_id, digest, "printing", False)
            existing_digest, state = row
            if existing_digest == digest and state == "failed":
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'printing', error = NULL, updated_at = ?,
                        title = COALESCE(title, ?), source = COALESCE(source, ?)
                    WHERE job_id = ? AND state = 'failed'
                    """,
                    (now, title, source, job_id),
                )
                connection.commit()
                return JobClaim(job_id, digest, "printing", False)
            connection.commit()
        if existing_digest != digest:
            return JobClaim(job_id, digest, "conflict", True)
        return JobClaim(job_id, digest, state, True)

    def finish(self, job_id: str, state: str, error: str | None = None) -> None:
        if state not in {"printed", "failed", "ambiguous"}:
            raise ValueError(f"invalid terminal print state: {state}")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET state = ?, error = ?, updated_at = ?
                WHERE job_id = ? AND state = 'printing'
                """,
                (state, error, time.time(), job_id),
            )


def default_journal_path() -> str:
    return os.getenv("RP_SERVE_STATE_PATH", "/tmp/receipt-print-serve.sqlite3")
