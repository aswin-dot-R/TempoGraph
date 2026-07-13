"""Streamlit-free rendering helpers shared by the Results UI and CLI tools.

Draws detection boxes, instance-mask overlays (decoded from the
``detections.mask_rle`` column) and depth heatmaps onto frames, and encodes
annotated strip videos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.rle import decode_from_string

PALETTE = [
    (66, 165, 245),
    (102, 187, 106),
    (239, 83, 80),
    (255, 167, 38),
    (171, 71, 188),
    (38, 198, 218),
    (236, 64, 122),
    (141, 110, 99),
]


def color_for(key: str, seen: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if key not in seen:
        seen[key] = PALETTE[len(seen) % len(PALETTE)]
    return seen[key]


def _entity_key(det: dict) -> str:
    """Stable per-entity colour key: track id when known, else class."""
    if det.get("track_id") is not None:
        return f"track_{det['track_id']}"
    return str(det.get("class_name", "?"))


def draw_detections(
    image_bgr: np.ndarray,
    dets: List[dict],
    min_conf: float = 0.0,
    highlight_classes: Optional[set] = None,
) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    seen: Dict[str, Tuple[int, int, int]] = {}
    for d in dets:
        if d["confidence"] < min_conf:
            continue
        cls = d["class_name"]
        color = color_for(cls, seen)
        if highlight_classes and cls not in highlight_classes:
            color = (120, 120, 120)
        x1 = max(0, int(d["x1"] * w))
        y1 = max(0, int(d["y1"] * h))
        x2 = min(w - 1, int(d["x2"] * w))
        y2 = min(h - 1, int(d["y2"] * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        depth_str = (
            f" d={d['mean_depth']:.2f}" if d.get("mean_depth") is not None else ""
        )
        label = f"{cls} {d['confidence']:.2f}{depth_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, max(th, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def draw_masks(
    image_bgr: np.ndarray, dets: List[dict], min_conf: float = 0.0, alpha: float = 0.45
) -> np.ndarray:
    """Overlay decoded instance masks as semi-transparent per-entity fills.

    Detections without a ``mask_rle`` value (bbox-only runs, legacy DBs) are
    skipped silently.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    seen: Dict[str, Tuple[int, int, int]] = {}
    for d in dets:
        rle = d.get("mask_rle")
        if not rle or d["confidence"] < min_conf:
            continue
        try:
            mask = decode_from_string(rle)
        except (ValueError, KeyError, TypeError):
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        color = color_for(_entity_key(d), seen)
        overlay = np.empty_like(img)
        overlay[:] = color
        sel = mask.astype(bool)
        img[sel] = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0.0)[sel]
    return img


def apply_depth_overlay(
    image_bgr: np.ndarray, depth_npy_path: Optional[str], alpha: float = 0.45
) -> np.ndarray:
    """Blend a depth heatmap (from a resolved .npy path) onto a frame."""
    if not depth_npy_path:
        return image_bgr
    p = Path(depth_npy_path)
    if not p.exists():
        return image_bgr
    try:
        depth = np.load(str(p))
    except Exception:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)


def build_annotated_video(
    frames: List[dict],
    det_by_frame: Dict[int, List[dict]],
    depth_by_frame: Dict[int, str],
    out_base: Path,
    fps: float,
    show_dets: bool = True,
    show_depth: bool = False,
    show_masks: bool = False,
    min_conf: float = 0.25,
    resolve: Optional[Callable[[str], Path]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Optional[Path]:
    """Encode the run's saved frames into an annotated strip video.

    ``out_base`` is the output path without extension; writes ``.mp4``
    (mp4v) with an ``.avi`` (MJPG) fallback. ``resolve`` maps stored image /
    depth paths to absolute paths (identity by default). Returns the written
    path or None on failure.
    """
    if not frames:
        return None
    _resolve = resolve or (lambda s: Path(s))
    sample = cv2.imread(str(_resolve(frames[0]["image_path"])))
    if sample is None:
        return None
    h, w = sample.shape[:2]

    out_path = out_base.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        out_path = out_base.with_suffix(".avi")
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
        )
        if not writer.isOpened():
            return None

    for i, fr in enumerate(frames):
        img = cv2.imread(str(_resolve(fr["image_path"])))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        if show_depth:
            depth_path = depth_by_frame.get(fr["frame_idx"])
            img = apply_depth_overlay(
                img, str(_resolve(depth_path)) if depth_path else None
            )
        dets = det_by_frame.get(fr["frame_idx"], [])
        if show_masks:
            img = draw_masks(img, dets, min_conf=min_conf)
        if show_dets:
            img = draw_detections(img, dets, min_conf=min_conf)
        ts_s = fr["timestamp_ms"] / 1000.0
        cv2.putText(
            img,
            f"frame #{fr['frame_idx']}  t={ts_s:.2f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"frame #{fr['frame_idx']}  t={ts_s:.2f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
        if on_progress is not None:
            on_progress((i + 1) / len(frames))
    writer.release()
    return out_path
