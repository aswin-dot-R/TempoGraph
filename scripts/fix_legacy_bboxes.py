#!/usr/bin/env python3
"""Fix legacy bboxes in pre-2026-04-27 TempoGraph runs.

Before 2026-04-27, bboxes were normalised against source video dimensions
instead of the saved JPEG dimensions. This script reads each run's
tempograph.db, determines the saved JPEG width/height from the first
frame, and rescales all detection coordinates.

Usage:
    python scripts/fix_legacy_bboxes.py --results-dir results/

    # Dry run (show what would change, don't write):
    python scripts/fix_legacy_bboxes.py --results-dir results/ --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2


def fix_run(run_dir: Path, dry_run: bool = False) -> dict:
    """Fix bboxes in a single run directory.

    Returns dict with stats: {"frames": N, "detections": N, "rescaled": bool}
    """
    db_path = run_dir / "tempograph.db"
    if not db_path.exists():
        return {"error": "no tempograph.db"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all frames
    frames = conn.execute(
        "SELECT frame_idx, image_path FROM frames ORDER BY frame_idx LIMIT 1"
    ).fetchall()
    if not frames:
        conn.close()
        return {"frames": 0, "detections": 0, "rescaled": False}

    # Read actual JPEG dimensions from first frame
    first_path = frames[0]["image_path"]
    p = Path(first_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    img = cv2.imread(str(p))
    if img is None:
        conn.close()
        return {"error": f"cannot read {p}"}
    jpeg_h, jpeg_w = img.shape[:2]

    # Check if bboxes look already normalised (values in [0, 1])
    sample_dets = conn.execute(
        "SELECT x1, y1, x2, y2 FROM detections LIMIT 10"
    ).fetchall()
    if not sample_dets:
        conn.close()
        return {"frames": 1, "detections": 0, "rescaled": False}

    # If max coord > 1.0, bboxes are in pixel space and need normalisation
    max_coord = max(
        max(d["x1"], d["y1"], d["x2"], d["y2"]) for d in sample_dets
    )
    if max_coord <= 1.0:
        conn.close()
        return {
            "frames": 1,
            "detections": len(sample_dets),
            "rescaled": False,
            "note": "already normalised (max coord <= 1.0)",
        }

    n_dets = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    if dry_run:
        conn.close()
        return {
            "detections": n_dets,
            "rescaled": False,
            "would_rescale": True,
            "jpeg_dims": f"{jpeg_w}x{jpeg_h}",
        }

    # Rescale: divide pixel coords by JPEG dimensions
    conn.execute(
        "UPDATE detections SET "
        "x1 = x1 / ?, y1 = y1 / ?, "
        "x2 = x2 / ?, y2 = y2 / ?",
        (jpeg_w, jpeg_h, jpeg_w, jpeg_h),
    )
    conn.commit()
    conn.close()

    return {
        "detections": n_dets,
        "rescaled": True,
        "jpeg_dims": f"{jpeg_w}x{jpeg_h}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix legacy bbox normalisation in old TempoGraph runs"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without modifying databases",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    runs = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and (p / "tempograph.db").exists()
    )
    print(f"Found {len(runs)} runs in {results_dir}")
    if args.dry_run:
        print("  (DRY RUN — no changes will be made)\n")

    for run_dir in runs:
        result = fix_run(run_dir, dry_run=args.dry_run)
        icon = "✓" if result.get("rescaled") else "·"
        if result.get("would_rescale"):
            icon = "→"
        if result.get("error"):
            icon = "✗"
        print(f"  {icon} {run_dir.name}: {result}")


if __name__ == "__main__":
    main()
