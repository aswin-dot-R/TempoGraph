"""Tests for the click-to-play video_player module.

All AppTest tests use the TEMPOGRAPH_RESULTS_DIR monkeypatch pattern from
tests/test_results_apptest.py — each test builds its own small run dir with
frames, a tiny mp4, and audio_segments, then patches the env var.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import TempoGraphDB

import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PAGE = REPO_ROOT / "ui" / "pages" / "Results.py"


# ─── helpers ──────────────────────────────────────────────────────────────────


def _tiny_jpeg(path: Path, seed: int = 0, h: int = 64, w: int = 128) -> Path:
    """Write a tiny random JPEG to ``path`` and return it."""
    rng = np.random.default_rng(seed)
    img = (rng.integers(0, 256, size=(h, w, 3))).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _tiny_mp4(
    path: Path, dur_s: float, fps: int = 10, h: int = 64, w: int = 128
) -> Path:
    """Write a tiny real mp4 with cv2.VideoWriter. Returns the path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n_frames = int(dur_s * fps)
    for _ in range(n_frames):
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        cv2.line(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
        cap.write(frame)
    cap.release()
    return path


def _make_fixture_run(
    results_dir: Path, name: str = "clickrun", video_dur_s: float = 10.0
) -> Path:
    """Build a small run dir: frames on disk, tiny mp4, audio segments."""
    import cv2

    run_dir = results_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    db = TempoGraphDB(run_dir / "tempograph.db")
    rng = np.random.default_rng(42)
    h, w = 64, 128
    for i, frame_idx in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]):
        img = (rng.integers(0, 256, size=(h, w, 3))).astype(np.uint8)
        p = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(p), img)
        db.insert_frame(
            frame_idx,
            frame_idx * video_dur_s / 10.0 * 1000,
            str(p),
            False,
            0.5,
        )

    # tiny mp4 for "original video" case
    video_path = run_dir / "original.mp4"
    _tiny_mp4(video_path, dur_s=video_dur_s, fps=10)
    db.set_meta("video_path", str(video_path))

    # audio segments
    for start_s, end_s, text in [
        (0.0, 1.5, "segment text one"),
        (1.5, 3.0, "segment text two"),
        (3.0, 4.5, "segment text three"),
        (4.5, 6.0, "segment text four"),
        (6.0, 7.5, "segment text five"),
        (7.5, 9.0, "segment text six"),
        (9.0, 10.5, "segment text seven"),
    ]:
        db.insert_audio_segment(
            start_ms=int(start_s * 1000),
            end_ms=int(end_s * 1000),
            text=text,
            no_speech_prob=0.01,
        )

    db.close()

    # analysis.json so the page has something to render
    analysis = {
        "entities": [],
        "visual_events": [],
        "audio_events": [],
        "summary": f"fixture run {name}",
    }
    (run_dir / "analysis.json").write_text(__import__("json").dumps(analysis))
    return run_dir


@pytest.fixture()
def fixture_results_dir(tmp_path, monkeypatch):
    """tmp_path with a sub-results dir containing one fixture run."""
    results_dir = tmp_path / "results"
    _make_fixture_run(results_dir)
    monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(results_dir))
    return results_dir


# ─── 1. make_strip_mapper math ────────────────────────────────────────────────


class TestMakeStripMapper:
    """Test make_strip_mapper() — the pure math contract."""

    def test_source_100_strip_25_zero(self):
        from ui.video_player import make_strip_mapper

        mapper = make_strip_mapper(100.0, 25.0)
        assert mapper(0.0) == 0.0
        assert mapper(40.0) == 10.0
        assert mapper(100.0) == 25.0
        assert mapper(200.0) == 25.0  # clamped

    def test_source_100_strip_25_negative(self):
        from ui.video_player import make_strip_mapper

        mapper = make_strip_mapper(100.0, 25.0)
        assert mapper(-5.0) == 0.0  # clamped to floor

    def test_source_zero_always_zero(self):
        from ui.video_player import make_strip_mapper

        mapper = make_strip_mapper(0.0, 25.0)
        assert mapper(0.0) == 0.0
        assert mapper(42.0) == 0.0
        assert mapper(-3.0) == 0.0

    def test_strip_longer_than_source_clamps_at_strip_end(self):
        from ui.video_player import make_strip_mapper

        # source 10 s, strip 100 s — mapper should clamp at strip_dur
        mapper = make_strip_mapper(10.0, 100.0)
        assert mapper(0.0) == 0.0
        assert mapper(10.0) == 100.0
        assert mapper(20.0) == 100.0  # clamped

    def test_linear_mapping(self):
        from ui.video_player import make_strip_mapper

        mapper = make_strip_mapper(10.0, 20.0)
        assert mapper(0.0) == 0.0
        assert mapper(5.0) == 10.0
        assert mapper(10.0) == 20.0


# ─── 2. resolve_video order ───────────────────────────────────────────────────


class TestResolveVideo:
    """Test resolve_video() — the run_meta resolution order."""

    def test_original_present_returns_original_kind(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_orig"
        run_dir.mkdir()
        video = run_dir / "clip.mp4"
        _tiny_mp4(video, dur_s=7.0, fps=10)
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.set_meta("video_path", str(video))
        db.close()

        result = resolve_video(run_dir, db)
        assert result is not None
        assert result.kind == "original"
        assert result.time_mapper(7.3) == 7.3

    def test_original_missing_falls_back_to_strip(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_strip"
        run_dir.mkdir()
        strip = run_dir / "annotated_strip.mp4"
        _tiny_mp4(strip, dur_s=5.0, fps=8)
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.set_meta("video_path", str(Path("/no/such/video.mp4")))
        db.close()

        result = resolve_video(run_dir, db)
        assert result is not None
        assert result.kind == "strip"
        # strip is 5s, so 3.0s source maps proportionally
        assert result.time_mapper(3.0) > 0.0
        assert result.time_mapper(3.0) < 5.0

    def test_original_deleted_falls_back_to_strip(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_del"
        run_dir.mkdir()
        video = run_dir / "clip.mp4"
        _tiny_mp4(video, dur_s=7.0, fps=10)
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.set_meta("video_path", str(video))
        db.close()
        # delete the video after writing meta (simulates missing original)
        video.unlink()
        # annotated strip must exist for the fallback to resolve
        strip = run_dir / "annotated_strip.mp4"
        _tiny_mp4(strip, dur_s=5.0, fps=8)

        result = resolve_video(run_dir, db)
        assert result is not None
        assert result.kind == "strip"

    def test_neither_original_nor_strip_returns_none(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_none"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.set_meta("video_path", str(Path("/no/such/video.mp4")))
        db.close()

        assert resolve_video(run_dir, db) is None

    def test_no_video_path_meta_no_exception(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_novideo"
        run_dir.mkdir()
        db = TempoGraphDB(run_dir / "tempograph.db")
        # no video_path meta at all
        db.close()

        assert resolve_video(run_dir, db) is None

    def test_strip_only_with_no_video_meta(self, tmp_path):
        from ui.video_player import resolve_video

        run_dir = tmp_path / "run_striponly"
        run_dir.mkdir()
        strip = run_dir / "annotated_strip.mp4"
        _tiny_mp4(strip, dur_s=4.0, fps=10)
        db = TempoGraphDB(run_dir / "tempograph.db")
        db.close()

        result = resolve_video(run_dir, db)
        assert result is not None
        assert result.kind == "strip"


# ─── AppTest tests ────────────────────────────────────────────────────────────


class TestClickToPlayAppTest:
    """Streamlit AppTest tests for Results.py click-to-play behavior."""

    def test_transcript_page_renders_with_segments(self, fixture_results_dir):
        """Page renders without exception and shows play_seg_ buttons."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception

        # at least 5 play_seg_ buttons (we have 7 segments)
        play_seg_btns = [
            b for b in at.button if b.key and b.key.startswith("play_seg_")
        ]
        assert len(play_seg_btns) >= 5

    def test_play_seg_click_sets_player_start_s(self, fixture_results_dir):
        """Clicking play_seg_{idx} sets session_state[player_start_s]."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception

        # segment with segment_id=2 starts at 1.5s (second inserted segment)
        at.button(key="play_seg_2").click().run()
        assert at.session_state["player_start_s"] == 1.5

    def test_frame_inspector_play_frame_sets_seconds(self, fixture_results_dir):
        """Clicking play_frame_{idx} sets player_start_s to timestamp_ms/1000."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception

        # frame 0 exists in fixture with timestamp_ms=0 → 0.0s
        at.button(key="play_frame_0").click().run()
        assert at.session_state["player_start_s"] == 0.0

    def test_pagination_shows_100_then_200(self, tmp_path, monkeypatch):
        """With 250 segments, initial render shows 100, after 'Show more' → 200."""
        from streamlit.testing.v1 import AppTest

        # Build a run dir with 250 audio segments
        results_dir = tmp_path / "results"
        run_dir = results_dir / "pagination_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        db = TempoGraphDB(run_dir / "tempograph.db")
        rng = np.random.default_rng(99)
        h, w = 64, 128
        for i in [0, 10]:
            img = (rng.integers(0, 256, size=(h, w, 3))).astype(np.uint8)
            p = frames_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(p), img)
            db.insert_frame(i, i * 1000, str(p), False, 0.5)

        # Insert 250 segments, each 1s long
        for seg_idx in range(250):
            db.insert_audio_segment(
                start_ms=seg_idx * 1000,
                end_ms=(seg_idx + 1) * 1000,
                text=f"segment text {seg_idx}",
                no_speech_prob=0.01,
            )

        db.close()
        analysis = {
            "entities": [],
            "visual_events": [],
            "audio_events": [],
            "summary": "pagination fixture",
        }
        (run_dir / "analysis.json").write_text(__import__("json").dumps(analysis))

        monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(results_dir))

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception

        initial_play_seg = [
            b for b in at.button if b.key and b.key.startswith("play_seg_")
        ]
        assert len(initial_play_seg) == 100

        # Click "Show more"
        show_more = at.button(key="show_more_transcript")
        assert show_more is not None
        show_more.click().run()

        # After show more: 200 buttons visible
        more_play_seg = [
            b for b in at.button if b.key and b.key.startswith("play_seg_")
        ]
        assert len(more_play_seg) == 200

    def test_no_video_page_renders_calmly(self, tmp_path, monkeypatch):
        """No video at all → page renders, no exception, click doesn't crash."""
        from streamlit.testing.v1 import AppTest

        # Build a run dir with NO video and NO strip
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        db = TempoGraphDB(run_dir / "tempograph.db")
        for i in [0, 10, 20]:
            img = (
                np.random.default_rng(i)
                .integers(0, 256, size=(64, 128, 3))
                .astype(np.uint8)
            )
            p = frames_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(p), img)
            db.insert_frame(i, i * 1000, str(p), False, 0.5)
        # No video_path meta, no annotated_strip.mp4
        db.close()
        analysis = {
            "entities": [],
            "visual_events": [],
            "audio_events": [],
            "summary": "empty run",
        }
        (run_dir / "analysis.json").write_text(__import__("json").dumps(analysis))

        monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(tmp_path))

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception

        # no play_seg_ buttons should crash — they may or may not exist
        # but clicking shouldn't raise
        for btn in at.button:
            if btn.key and btn.key.startswith("play_seg_"):
                try:
                    btn.click().run()
                except Exception:
                    pytest.fail(
                        f"play_seg_{btn.key[len('play_seg_:'):] or 'x'} click "
                        f"raised an exception"
                    )
