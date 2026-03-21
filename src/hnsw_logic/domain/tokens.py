"""Tokenization and vector similarity helpers."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    """Split text into normalized lowercase tokens."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def deterministic_vector(text: str, dim: int) -> np.ndarray:
    """Build a deterministic sparse vector for lightweight tests and stubs."""

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
    """Return cosine similarity for two vectors."""

    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(a, b) / denom)


def top_terms(text: str, limit: int = 8) -> list[str]:
    """Return the most frequent informative terms in the given text."""

    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]
