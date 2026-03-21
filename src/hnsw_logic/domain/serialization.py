"""Serialization and filesystem helpers used across the application."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson


def utc_now() -> str:
    """Return the current UTC timestamp in canonical ISO-8601 form."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    """Create a directory tree if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file and return a default value when it is missing."""

    if not path.exists():
        return default
    return orjson.loads(path.read_bytes())


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload using stable formatting and key ordering."""

    ensure_dir(path.parent)
    path.write_bytes(orjson.dumps(to_jsonable(payload), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def append_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Append rows to a JSONL file, creating parent directories when needed."""

    ensure_dir(path.parent)
    with path.open("ab") as handle:
        for row in rows:
            handle.write(orjson.dumps(to_jsonable(row)))
            handle.write(b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(orjson.loads(line))
    return rows


def to_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-serializable structures."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
