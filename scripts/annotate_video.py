#!/usr/bin/env python3
"""Render a run's annotated strip video from the command line.

Same code path as the Results page "Annotated video" tab
(src/annotate.build_annotated_video), including the --masks overlay for
seg-model runs whose detections carry mask_rle.

Usage:
    python3 scripts/annotate_video.py results/<run> [--masks] [--depth]
        [--fps 4] [--min-conf 0.25] [--no-dets] [-o out_base]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.annotate import build_annotated_video  # noqa: E402


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode an annotated strip video for a TempoGraph run."
    )
    parser.add_argument("run_dir", help="run directory containing tempograph.db")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--masks", action="store_true",
                        help="overlay instance masks decoded from mask_rle "
                             "(seg-model runs)")
    parser.add_argument("--depth", action="store_true",
                        help="overlay depth heatmaps where available")
    parser.add_argument("--no-dets", action="store_true",
                        help="skip bounding boxes")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("-o", "--out-base", default=None,
                        help="output path without extension "
                             "(default: <run_dir>/annotated_strip)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    db_path = run_dir / "tempograph.db"
    if not db_path.exists():
        print(f"error: {db_path} not found", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    frames = [dict(r) for r in conn.execute(
        "SELECT frame_idx, timestamp_ms, image_path, is_keyframe, delta_score "
        "FROM frames ORDER BY frame_idx ASC")]
    dets = [dict(r) for r in conn.execute("SELECT * FROM detections")]
    depth_rows = [dict(r) for r in conn.execute(
        "SELECT frame_idx, depth_npy_path FROM depth_frames")]
    conn.close()

    det_by_frame = {}
    for d in dets:
        det_by_frame.setdefault(d["frame_idx"], []).append(d)
    depth_by_frame = {r["frame_idx"]: r["depth_npy_path"] for r in depth_rows}

    out_base = Path(args.out_base) if args.out_base else run_dir / "annotated_strip"
    out = build_annotated_video(
        frames=frames,
        det_by_frame=det_by_frame,
        depth_by_frame=depth_by_frame,
        out_base=out_base,
        fps=args.fps,
        show_dets=not args.no_dets,
        show_depth=args.depth,
        show_masks=args.masks,
        min_conf=args.min_conf,
        resolve=_resolve,
    )
    if out is None:
        print("error: failed to encode annotated video", file=sys.stderr)
        return 1
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
