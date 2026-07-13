"""Tests for src/highlight_reel.py — greedy frame selection & ffmpeg reel export."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.highlight_reel import (
    build_highlight_reel,
    pick_highlight_spans,
)
from src.storage import TempoGraphDB

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(out.stdout.strip())


def make_highlight_run(tmp_path: Path, delta_scores=None) -> Path:
    """Create a run dir with TempoGraphDB frames at 1s intervals."""
    run_dir = tmp_path / "highlightrun"
    run_dir.mkdir()
    db = TempoGraphDB(run_dir / "tempograph.db")
    if delta_scores is None:
        delta_scores = [9, 1, 8, 1, 7, 1, 5, 1, 4, 1]
    for i, score in enumerate(delta_scores):
        db.insert_frame(
            i * 30, i * 1000, f"frames/frame_{i*30:06d}.jpg", False, float(score)
        )
    db.close()
    return run_dir


def make_test_video(tmp_path: Path, duration_s: float = 14.0) -> Path:
    """Generate a testsrc video via ffmpeg."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "source14.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size=320x180:rate=30",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


# ─── pure span selection ──────────────────────────────────────────────────────


class TestPickHighlightSpans:
    def test_greedy_pick_hand_checked(self, tmp_path):
        """10 frames at 1s intervals, delta_scores [9,1,8,1,7,1,5,1,4,1].

        Hand-computed expected spans (padding 1.5s, min_gap 3s):
          Greedy by score desc checks padded span vs. every accepted span:
            t=0 (score=9) → no accepted spans → accept → (0, 1500)
            t=2 (score=8) → padded (500, 3500) overlaps (0, 1500) → reject
            t=4 (score=7) → padded (2500, 5500) gap from (0,1500) = 1000ms < 3000 → reject
            t=6 (score=5) → padded (4500, 7500) gap from (0,1500) = 3000ms >= 3000 → accept → (4500, 7500)
            t=8 (score=4) → padded (6500, 8500) overlaps (4500, 7500) → reject
            t=1..9 (score=1) → all too close to an accepted span → reject
          Final: [(0, 1500), (4500, 7500)] — total 4.5s < 6s target.
        """
        run_dir = make_highlight_run(tmp_path)
        spans = pick_highlight_spans(
            run_dir,
            target_duration_s=6.0,
            min_gap_s=3.0,
            span_padding_s=1.5,
        )
        assert spans == [(0, 1500), (4500, 7500)]

    def test_min_gap_respected(self, tmp_path):
        """Two top frames 1s apart → only the higher-scored one accepted."""
        run_dir = tmp_path / "mingaprun"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.insert_frame(0, 0, "frames/0.jpg", False, 10.0)
        db.insert_frame(1, 1000, "frames/1.jpg", False, 9.0)
        db.close()
        spans = pick_highlight_spans(
            run_dir,
            target_duration_s=60.0,
            min_gap_s=3.0,
            span_padding_s=1.5,
        )
        assert len(spans) == 1
        assert spans[0] == (0, 1500)  # only the higher-scored frame

    def test_merge_overlapping_padded_spans(self, tmp_path):
        """Top frames 2s apart → padded spans overlap → only the higher-scored one accepted (greedy rejects overlap)."""
        run_dir = tmp_path / "mergeonrun"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.insert_frame(0, 0, "frames/0.jpg", False, 10.0)
        db.insert_frame(1, 2000, "frames/1.jpg", False, 9.0)
        db.close()
        spans = pick_highlight_spans(
            run_dir,
            target_duration_s=60.0,
            min_gap_s=3.0,
            span_padding_s=1.5,
        )
        # Padded spans: (0, 1500) and (500, 3500) overlap → greedy rejects second
        assert len(spans) == 1
        assert spans[0] == (0, 1500)

    def test_target_duration_limits_accumulation(self, tmp_path):
        """Total accepted span time ≤ target + one span's worth (greedy overshoot)."""
        run_dir = tmp_path / "targetrun"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        # 5 frames at 1s intervals, all score 10 → greedy picks t=0, then skips
        # close frames; total should not exceed target + padding
        for i in range(5):
            db.insert_frame(i * 30, i * 1000, f"frames/{i*30:06d}.jpg", False, 10.0)
        db.close()
        spans = pick_highlight_spans(
            run_dir,
            target_duration_s=3.0,
            min_gap_s=3.0,
            span_padding_s=1.5,
        )
        total_s = sum((e - s) for s, e in spans) / 1000.0
        assert total_s <= 3.0 + 1.5  # target + one span's worth

    def test_empty_db_returns_empty(self, tmp_path):
        """Empty DB → no spans."""
        run_dir = tmp_path / "emptyrun"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.close()
        spans = pick_highlight_spans(run_dir)
        assert spans == []

    def test_single_frame_db(self, tmp_path):
        """Single-frame DB → one clamped span."""
        run_dir = tmp_path / "singleframe"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.insert_frame(0, 0, "frames/0.jpg", False, 5.0)
        db.close()
        spans = pick_highlight_spans(run_dir)
        assert spans == [(0, 1500)]  # clamped at 0

    def test_single_span_no_fade(self, tmp_path):
        """Single span, no fade — output duration ≈ span duration."""
        video = make_test_video(tmp_path / "single")
        spans = [(5000, 7000)]  # 2s span
        out_path = tmp_path / "out_single.mp4"
        result = build_highlight_reel(video, spans, out_path)
        assert result == out_path
        got = _ffprobe_duration(out_path)
        assert got is not None
        assert abs(got - 2.0) <= 0.5

    def test_empty_spans_raises(self, tmp_path):
        """Empty spans → ValueError."""
        video = make_test_video(tmp_path / "empty")
        with pytest.raises(ValueError):
            build_highlight_reel(video, [], tmp_path / "out.mp4")

    def test_nonexistent_video_raises(self, tmp_path):
        """Nonexistent video → RuntimeError."""
        with pytest.raises(RuntimeError):
            build_highlight_reel(
                tmp_path / "nope.mp4", [(0, 1000)], tmp_path / "out.mp4"
            )


# ─── ffmpeg integration ───────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
class TestBuildHighlightReel:
    def test_ffmpeg_integration(self, tmp_path_factory):
        """Real testsrc video + 2 hand spans → output exists, h264, duration valid."""
        src = make_test_video(tmp_path_factory.mktemp("src"), duration_s=14.0)
        out = tmp_path_factory.mktemp("out") / "reel.mp4"
        spans = [(2000, 4000), (7000, 9000)]  # 2s + 2s = 4s
        result = build_highlight_reel(src, spans, out)
        assert result.exists()
        dur = _ffprobe_duration(result)
        assert dur is not None
        # sum of spans minus one crossfade (spans are in ms, dur in s)
        want = sum((e - s) for s, e in spans) / 1000.0 - 0.25
        assert abs(dur - want) <= 0.5
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(result),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert probe.stdout.strip() == "h264"
