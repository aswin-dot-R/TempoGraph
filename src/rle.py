"""COCO-style run-length encoding for binary instance masks.

Pure helpers (numpy only — no new dependencies). The encoding follows the
COCO "uncompressed RLE" convention:

- the mask is flattened in **column-major (Fortran) order**;
- ``counts`` alternates run lengths of 0s and 1s, always starting with the
  number of leading zeros (which may be 0 when the mask starts with a 1);
- ``size`` is ``[height, width]``.

The JSON string form (``encode_to_string`` / ``decode_from_string``) is what
gets persisted in the ``detections.mask_rle`` column.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np


def encode(mask) -> Dict:
    """Encode a 2-D binary mask into a COCO-style uncompressed RLE dict.

    Any array-like of shape (H, W) is accepted; nonzero values count as
    foreground. Returns ``{"size": [H, W], "counts": [int, ...]}``.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    h, w = arr.shape
    flat = (arr.flatten(order="F") != 0).astype(np.uint8)
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": []}

    change = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(boundaries).tolist()
    counts: List[int] = [int(r) for r in runs]
    if flat[0] == 1:
        # COCO counts always start with the number of zeros.
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": counts}


def decode(rle: Dict) -> np.ndarray:
    """Decode a COCO-style uncompressed RLE dict into a uint8 (H, W) mask."""
    h, w = (int(v) for v in rle["size"])
    counts = rle.get("counts", [])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        c = int(c)
        if c < 0:
            raise ValueError(f"negative run length {c} in RLE counts")
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    if pos != h * w:
        raise ValueError(
            f"RLE counts sum to {pos}, expected {h * w} for size [{h}, {w}]"
        )
    return flat.reshape((h, w), order="F")


def encode_to_string(mask) -> str:
    """Encode a binary mask to the JSON string stored in ``mask_rle``."""
    return json.dumps(encode(mask), separators=(",", ":"))


def decode_from_string(s: str) -> np.ndarray:
    """Decode a ``mask_rle`` JSON string back into a uint8 (H, W) mask."""
    return decode(json.loads(s))
