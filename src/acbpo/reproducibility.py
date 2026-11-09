"""Deterministic helpers shared by release analyses."""

from __future__ import annotations

from hashlib import sha256


def stable_seed(namespace: str, *parts: object) -> int:
    """Return a platform-independent uint32 seed for a named analysis stream."""
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)
