from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return orjson.loads(path.read_bytes())


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_bytes(orjson.dumps(to_jsonable(payload), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def append_jsonl(path: Path, rows: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("ab") as handle:
        for row in rows:
            handle.write(orjson.dumps(to_jsonable(row)))
            handle.write(b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(orjson.loads(line))
    return rows


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
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


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def deterministic_vector(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(a, b) / denom)


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]
