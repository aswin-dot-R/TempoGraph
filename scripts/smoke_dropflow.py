"""CPU end-to-end smoke test for the dropflow UI pipeline.

Per PS acceptance item 4:
  ffmpeg -y -f lavfi -i testsrc=duration=5:size=640x360:rate=10 /tmp/tg_smoke.mp4
  $PY scripts/smoke_dropflow.py /tmp/tg_smoke.mp4

The script uses auto_profile.probe() + derive_plan() to derive the plan
from the clip, then forces device=cpu, vlm=skip, depth=skip, audio=skip.

Asserts: exit 0, run DB exists, frames>0 rows, detections table exists.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: smoke_dropflow.py <video_path>")
        print("Create a test video first:")
        print('  ffmpeg -y -f lavfi -i testsrc=duration=5:size=640x360:rate=10 /tmp/tg_smoke.mp4')
        return 1

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    # Derive the plan from the video using auto_profile
    sys.path.insert(0, str(REPO_ROOT))
    from src.auto_profile import probe, derive_plan  # noqa: E402

    try:
        facts = probe(str(video_path))
        plan = derive_plan(facts)
        print(f"Probed: {facts.duration_s:.0f}s, {facts.width}x{facts.height}, "
              f"fps={facts.fps}, audio={facts.has_audio}")
        print(f"Derived: YOLO{plan.yolo_size}-seg @ {plan.yolo_fps} fps, "
              f"VLM fps={plan.vlm_fps}, chunk={plan.chunk_size}")
    except Exception as e:
        print(f"auto_profile failed: {e}")
        # Fall back to defaults
        video_path = str(video_path)
        plan = None

    tmp = tempfile.mkdtemp(prefix="tg_smoke_")
    output_dir = Path(tmp) / "output"

    # Build CLI args from derived plan, but force CPU + skip VLM/depth/audio
    cmd = [
        PYTHON,
        "-m",
        "src.pipeline_v2",
        "--video",
        str(video_path),
        "--output",
        str(output_dir),
        "--skip-vlm",
        "--camera",
        "static",
        "--confidence",
        "0.5",
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    print(f"Pipeline exit code: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")

    try:
        assert result.returncode == 0, (
            f"Pipeline exited with code {result.returncode}"
        )
        assert output_dir.exists(), f"Output dir missing: {output_dir}"

        db_path = output_dir / "tempograph.db"
        assert db_path.exists(), f"DB missing: {db_path}"

        with sqlite3.connect(str(db_path)) as conn:
            frames_rows = conn.execute(
                "SELECT COUNT(*) FROM frames"
            ).fetchone()[0]
            has_detections = (
                conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='detections'"
                ).fetchone()
                is not None
            )

        assert int(frames_rows) > 0, f"Expected frames>0, got {frames_rows}"
        assert has_detections, "detections table does not exist"

        print(
            f"PASS: {int(frames_rows)} frames, "
            f"detections table exists"
        )
        return 0

    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
