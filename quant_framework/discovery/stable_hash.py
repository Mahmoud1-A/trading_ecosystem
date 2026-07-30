"""Process-stable integer hashing for research reproducibility.

Builtin ``hash()`` is salted per process (unless PYTHONHASHSEED is fixed).
Research paths must use this helper so FamilySpecs, seeds, and fingerprints
match across separate Python processes.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_int_hash(value: str | bytes | int | float | Any, *, bits: int = 64) -> int:
    """Deterministic non-negative integer from a value via SHA-256."""
    if bits not in (32, 64, 128):
        raise ValueError(f"bits must be 32, 64, or 128; got {bits}")
    if isinstance(value, bytes):
        payload = value
    else:
        payload = str(value).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    n_bytes = bits // 8
    return int.from_bytes(digest[:n_bytes], "big", signed=False)


def stable_seed(*parts: Any, salt: int = 0) -> int:
    """Combine parts into a 31-bit numpy-friendly seed."""
    joined = "|".join(str(p) for p in parts) + f"|{salt}"
    return int(stable_int_hash(joined, bits=64) % (2**31 - 1))
