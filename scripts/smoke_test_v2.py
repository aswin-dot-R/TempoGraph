"""End-to-end smoke test for TempoGraph v2 without llama-server."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "sample.mp4"


@dataclass
class PassResult:
    name: str
    status: str
    elapsed_s: float
    frames_rows: int
    detection_rows: int
    depth_rows: int
    jpg_count: int
    notes: str = ""


def ensure_fixture() -> None:
    if FIXTURE_PATH.exists():
        return

    cmd = [sys.executable, "tools/make_test_video.py"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0 or not FIXTURE_PATH.exists():
        raise RuntimeError(
            "Failed to create synthetic fixture.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def run_pipeline_pass(name: str, extra_args: list[str], output_dir: Path) -> PassResult:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.pipeline_v2",
        "--video",
        str(FIXTURE_PATH.relative_to(REPO_ROOT)),
        "--output",
        str(output_dir.relative_to(REPO_ROOT)),
        "--skip-vlm",
        *extra_args,
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    elapsed_s = time.perf_counter() - start

    try:
        verify_outputs(output_dir, expect_depth="--depth" in extra_args)
        status = "PASS"
        notes = ""
    except AssertionError as exc:
        status = "FAIL"
        notes = str(exc)

    if result.returncode != 0 and status == "PASS":
        status = "FAIL"
        notes = (
            f"Pipeline exited with code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    frames_rows, detection_rows, depth_rows = read_db_counts(output_dir / "tempograph.db")
    jpg_count = len(list((output_dir / "frames").glob("*.jpg")))

    if detection_rows == 0 and status == "PASS":
        notes = "YOLO produced 0 detections on synthetic shapes; pipeline still passed."

    return PassResult(
        name=name,
        status=status,
        elapsed_s=elapsed_s,
        frames_rows=frames_rows,
        detection_rows=detection_rows,
        depth_rows=depth_rows,
        jpg_count=jpg_count,
        notes=notes,
    )


def verify_outputs(output_dir: Path, expect_depth: bool) -> None:
    db_path = output_dir / "tempograph.db"
    assert db_path.exists(), f"Missing SQLite database: {db_path}"

    frames_rows, detection_rows, depth_rows = read_db_counts(db_path)
    assert frames_rows >= 1, "frames table is empty"
    assert table_exists(db_path, "detections"), "detections table does not exist"
    assert table_exists(db_path, "depth_frames"), "depth_frames table does not exist"

    frames_dir = output_dir / "frames"
    assert frames_dir.exists(), f"Missing frames directory: {frames_dir}"
    jpg_count = len(list(frames_dir.glob("*.jpg")))
    assert jpg_count == frames_rows, (
        f"JPEG count mismatch: {jpg_count} files vs {frames_rows} DB rows"
    )

    analysis_path = output_dir / "analysis.json"
    assert not analysis_path.exists(), "analysis.json should not exist when --skip-vlm is used"

    if expect_depth:
        depth_dir = output_dir / "depth"
        npy_files = list(depth_dir.glob("*.npy"))
        assert depth_rows >= 1, "depth_frames table is empty despite --depth"
        assert npy_files, f"No .npy files found in {depth_dir}"
    else:
        assert depth_rows == 0, "depth_frames should be empty when --depth is not enabled"

    # detections rows may legitimately be zero on synthetic content; existence is enough.
    _ = detection_rows


def read_db_counts(db_path: Path) -> tuple[int, int, int]:
    with sqlite3.connect(db_path) as conn:
        frames_rows = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        detection_rows = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        depth_rows = conn.execute("SELECT COUNT(*) FROM depth_frames").fetchone()[0]
    return int(frames_rows), int(detection_rows), int(depth_rows)


def table_exists(db_path: Path, table_name: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def print_summary(results: list[PassResult]) -> None:
    headers = ["pass", "status", "elapsed_s", "frames", "detections", "depth_rows", "jpgs"]
    rows = [
        [
            res.name,
            res.status,
            f"{res.elapsed_s:.2f}",
            str(res.frames_rows),
            str(res.detection_rows),
            str(res.depth_rows),
            str(res.jpg_count),
        ]
        for res in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))

    print()
    for res in results:
        if res.notes:
            print(f"[{res.name}] {res.notes}")


def main() -> int:
    ensure_fixture()

    passes = [
        ("default_skip_vlm", [], REPO_ROOT / "results" / "smoke_test"),
        ("depth_static_skip_vlm", ["--depth", "--camera", "static"], REPO_ROOT / "results" / "smoke_test_depth"),
        ("moving_skip_vlm", ["--camera", "moving"], REPO_ROOT / "results" / "smoke_test_moving"),
    ]

    results: list[PassResult] = []
    for name, extra_args, output_dir in passes:
        results.append(run_pipeline_pass(name, extra_args, output_dir))

    print_summary(results)

    failed = [res for res in results if res.status != "PASS"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
