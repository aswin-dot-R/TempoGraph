"""CPU end-to-end smoke test for the dropflow UI pipeline.

Runs the pipeline on a 5-second synthetic video with:
- device=cpu
- vlm=skip
- depth=skip
- audio=skip

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

# Use the system python which has ultralytics (anaconda3 may not)
PYTHON = sys.executable


def main() -> int:
    # Create a 5s synthetic video
    tmp = tempfile.mkdtemp(prefix="tg_smoke_")
    video_path = Path(tmp) / "tg_smoke.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=5:size=640x360:rate=10",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"ffmpeg failed: {result.stderr}")
        return 1

    if not video_path.exists():
        print("Synthetic video not created")
        return 1

    output_dir = Path(tmp) / "output"

    # Run the pipeline via CLI
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
        "--yolo-size",
        "n",
        "--yolo-fps",
        "1.0",
        "--confidence",
        "0.5",
    ]

    env = dict(os.environ)
    # Force CPU mode
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    exit_code = result.returncode
    print(f"Pipeline exit code: {exit_code}")
    if result.stdout:
        print(f"stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")

    try:
        # Assertions
        assert exit_code == 0, f"Pipeline exited with code {exit_code}"
        assert output_dir.exists(), f"Output dir missing: {output_dir}"

        db_path = output_dir / "tempograph.db"
        assert db_path.exists(), f"DB missing: {db_path}"

        with sqlite3.connect(str(db_path)) as conn:
            frames_rows = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            has_detections = (
                conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='detections'"
                ).fetchone()
                is not None
            )

        assert int(frames_rows) > 0, f"Expected frames>0, got {frames_rows}"
        assert has_detections, "detections table does not exist"

        print(f"PASS: {int(frames_rows)} frames, detections table exists")
        return 0

    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
