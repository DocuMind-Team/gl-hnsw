from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

import orjson

from hnsw_logic.core.models import JobStatus
from hnsw_logic.core.utils import ensure_dir, utc_now


class JobRegistry:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists jobs (
                    job_id text primary key,
                    job_type text not null,
                    state text not null,
                    payload text not null,
                    message text not null,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )

    @staticmethod
    def _serialize_payload(payload: str | dict[str, Any]) -> str:
        if isinstance(payload, str):
            return payload
        return orjson.dumps(payload).decode("utf-8")

    @staticmethod
    def _deserialize_payload(payload: str) -> dict[str, Any]:
        try:
            parsed = orjson.loads(payload)
        except orjson.JSONDecodeError:
            return {"raw": payload}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def create(self, job_type: str, payload: str | dict[str, Any]) -> JobStatus:
        now = utc_now()
        payload_text = self._serialize_payload(payload)
        job = JobStatus(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            state="queued",
            payload=self._deserialize_payload(payload_text),
            message="queued",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                "insert into jobs values (?, ?, ?, ?, ?, ?, ?)",
                (job.job_id, job.job_type, job.state, payload_text, job.message, job.created_at, job.updated_at),
            )
        return job

    def update(self, job_id: str, state: str, message: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "update jobs set state = ?, message = ?, updated_at = ? where job_id = ?",
                (state, message, now, job_id),
            )

    def get(self, job_id: str) -> JobStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "select job_id, job_type, state, payload, message, created_at, updated_at from jobs where job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return JobStatus(
            job_id=row[0],
            job_type=row[1],
            state=row[2],
            payload=self._deserialize_payload(row[3]),
            message=row[4],
            created_at=row[5],
            updated_at=row[6],
        )

    def recent(self, limit: int = 10) -> list[JobStatus]:
        with self._connect() as conn:
            rows = conn.execute(
                "select job_id, job_type, state, payload, message, created_at, updated_at from jobs order by updated_at desc limit ?",
                (limit,),
            ).fetchall()
        return [
            JobStatus(
                job_id=row[0],
                job_type=row[1],
                state=row[2],
                payload=self._deserialize_payload(row[3]),
                message=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]
