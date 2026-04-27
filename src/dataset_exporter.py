"""Export TempoGraph pipeline results to standard ML dataset formats.

Reads from the per-run SQLite database (tempograph.db) and analysis.json
to produce:
  - COCO-format detection annotations (bbox JSON)
  - Dense video captioning JSONL (ActivityNet-Captions style)
  - Per-frame caption JSONL for VLM fine-tuning
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── COCO bbox export ───────────────────────────────────────────────


def export_coco_annotations(
    db_path: Path,
    output_path: Path,
    min_confidence: float = 0.0,
) -> Dict:
    """Export YOLO detections from a TempoGraph SQLite DB to COCO format.

    Args:
        db_path: Path to tempograph.db.
        output_path: Where to write coco_annotations.json.
        min_confidence: Minimum detection confidence to include.

    Returns:
        The COCO-format dict that was written.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Build categories from unique class names
    classes = conn.execute(
        "SELECT DISTINCT class_name FROM detections ORDER BY class_name"
    ).fetchall()
    class_to_id: Dict[str, int] = {}
    categories = []
    for i, row in enumerate(classes, start=1):
        name = row["class_name"]
        class_to_id[name] = i
        categories.append({"id": i, "name": name, "supercategory": "object"})

    # Build images from frames table
    frames = conn.execute(
        "SELECT frame_idx, timestamp_ms, image_path FROM frames "
        "ORDER BY frame_idx"
    ).fetchall()

    # Read dimensions from first image only — all frames are resized to
    # the same frame_max_width by the pipeline, so they share dimensions.
    default_w, default_h = 640, 360
    if frames:
        import cv2
        first_img = cv2.imread(frames[0]["image_path"])
        if first_img is not None:
            default_h, default_w = first_img.shape[:2]

    images = []
    frame_idx_to_image_id: Dict[int, int] = {}
    for img_id, row in enumerate(frames):
        frame_idx_to_image_id[row["frame_idx"]] = img_id
        images.append({
            "id": img_id,
            "file_name": Path(row["image_path"]).name,
            "width": default_w,
            "height": default_h,
            "frame_idx": row["frame_idx"],
            "timestamp_ms": row["timestamp_ms"],
        })

    # Build annotations from detections
    detections = conn.execute(
        "SELECT * FROM detections WHERE confidence >= ? ORDER BY detection_id",
        (min_confidence,),
    ).fetchall()

    annotations = []
    for det in detections:
        frame_idx = det["frame_idx"]
        if frame_idx not in frame_idx_to_image_id:
            continue

        image_id = frame_idx_to_image_id[frame_idx]
        img_info = images[image_id]
        w, h = img_info["width"], img_info["height"]

        # Convert normalised [0,1] coords to pixel coords
        x1_px = det["x1"] * w
        y1_px = det["y1"] * h
        x2_px = det["x2"] * w
        y2_px = det["y2"] * h
        bbox_w = x2_px - x1_px
        bbox_h = y2_px - y1_px

        cat_id = class_to_id.get(det["class_name"], 0)
        annotations.append({
            "id": det["detection_id"],
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [round(x1_px, 1), round(y1_px, 1),
                     round(bbox_w, 1), round(bbox_h, 1)],
            "area": round(bbox_w * bbox_h, 1),
            "iscrowd": 0,
            "confidence": round(det["confidence"], 4),
            "mean_depth": det["mean_depth"],
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"COCO export: {len(images)} images, "
        f"{len(annotations)} annotations, "
        f"{len(categories)} categories → {output_path}"
    )
    conn.close()
    return coco


# ── Dense captioning JSONL export ──────────────────────────────────


def export_captions_jsonl(
    chunks_path: Path,
    analysis_path: Path,
    output_path: Path,
    video_name: Optional[str] = None,
) -> int:
    """Export per-chunk and per-event captions as JSONL.

    Produces two styles of entries:
      1. Per-chunk dense captions (from chunks.json)
      2. Per-event temporal grounding (from analysis.json visual_events)

    Args:
        chunks_path: Path to chunks.json.
        analysis_path: Path to analysis.json.
        output_path: Where to write the JSONL file.
        video_name: Optional video identifier.

    Returns:
        Number of lines written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as out:
        # Per-chunk captions
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            for c in chunks:
                if not c.get("summary"):
                    continue
                frame_indices = c.get("frame_indices", [])
                entry = {
                    "type": "chunk_caption",
                    "video": video_name or "unknown",
                    "chunk_id": c["chunk_id"],
                    "frame_indices": frame_indices,
                    "caption": c["summary"],
                    "per_frame": c.get("per_frame_lines", {}),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        # Per-event temporal grounding
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)

            for i, ev in enumerate(analysis.get("visual_events", [])):
                entry = {
                    "type": "visual_event",
                    "video": video_name or "unknown",
                    "event_idx": i,
                    "event_type": ev.get("type", "other"),
                    "entities": ev.get("entities", []),
                    "start_time": ev.get("start_time", "00:00"),
                    "end_time": ev.get("end_time", "00:00"),
                    "description": ev.get("description", ""),
                    "confidence": ev.get("confidence", 0.0),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

            for i, ae in enumerate(analysis.get("audio_events", [])):
                entry = {
                    "type": "audio_event",
                    "video": video_name or "unknown",
                    "event_idx": i,
                    "start_time": ae.get("start_time", "00:00"),
                    "end_time": ae.get("end_time", "00:00"),
                    "text": ae.get("text", ""),
                    "speaker": ae.get("speaker", "unknown"),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

            for i, mc in enumerate(analysis.get("multimodal_correlations", [])):
                entry = {
                    "type": "multimodal_correlation",
                    "video": video_name or "unknown",
                    "correlation_idx": i,
                    "audio_idx": mc.get("audio_idx"),
                    "visual_idx": mc.get("visual_idx"),
                    "description": mc.get("description", ""),
                    "confidence": mc.get("confidence", 0.0),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Captions JSONL: {count} entries → {output_path}")
    return count


# ── Per-frame VLM fine-tuning JSONL ────────────────────────────────


def export_frame_captions_jsonl(
    chunks_path: Path,
    db_path: Path,
    output_path: Path,
    video_name: Optional[str] = None,
) -> int:
    """Export per-frame (image_path, caption) pairs for VLM fine-tuning.

    One JSONL line per frame that has a non-trivial caption (excludes
    '(no change)' entries).

    Args:
        chunks_path: Path to chunks.json.
        db_path: Path to tempograph.db.
        output_path: Where to write the JSONL file.
        video_name: Optional video identifier.

    Returns:
        Number of lines written.
    """
    if not chunks_path.exists():
        logger.warning(f"No chunks.json at {chunks_path}")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    with open(chunks_path) as f:
        chunks = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as out:
        for c in chunks:
            for fidx_str, caption in c.get("per_frame_lines", {}).items():
                if not caption or caption.strip().lower() == "(no change)":
                    continue
                fidx = int(fidx_str)
                row = conn.execute(
                    "SELECT image_path, timestamp_ms FROM frames WHERE frame_idx = ?",
                    (fidx,),
                ).fetchone()
                if not row:
                    continue

                entry = {
                    "video": video_name or "unknown",
                    "frame_idx": fidx,
                    "image_path": row["image_path"],
                    "timestamp_ms": row["timestamp_ms"],
                    "caption": caption,
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    conn.close()
    logger.info(f"Frame captions JSONL: {count} entries → {output_path}")
    return count


# ── Convenience: export all formats for a single run ───────────────


def export_all(
    run_dir: Path,
    video_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Export all dataset formats for a single pipeline run.

    Args:
        run_dir: Path to the results/<video>/ directory.
        video_name: Optional human-readable video name.
        output_dir: Where to write exports (default: run_dir/exports/).

    Returns:
        Dict mapping format name to output file path.
    """
    if output_dir is None:
        output_dir = run_dir / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = run_dir / "tempograph.db"
    chunks_path = run_dir / "chunks.json"
    analysis_path = run_dir / "analysis.json"

    outputs: Dict[str, Path] = {}

    if db_path.exists():
        coco_path = output_dir / "coco_annotations.json"
        export_coco_annotations(db_path, coco_path)
        outputs["coco"] = coco_path

    if chunks_path.exists() or analysis_path.exists():
        captions_path = output_dir / "captions.jsonl"
        export_captions_jsonl(chunks_path, analysis_path, captions_path, video_name)
        outputs["captions"] = captions_path

    if chunks_path.exists() and db_path.exists():
        frames_path = output_dir / "frame_captions.jsonl"
        export_frame_captions_jsonl(chunks_path, db_path, frames_path, video_name)
        outputs["frame_captions"] = frames_path

    logger.info(f"All exports complete: {list(outputs.keys())}")
    return outputs
