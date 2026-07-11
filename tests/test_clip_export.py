"""Span math, event selection and ffmpeg export tests for src/clip_export.py."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_export import (
    _ffprobe_duration,
    export_clips,
    pad_and_merge,
    select_events,
)
from src.storage import TempoGraphDB

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ─── pure span math ───────────────────────────────────────────────────────────

class TestPadAndMerge:
    def test_empty(self):
        assert pad_and_merge([]) == []

    def test_single_event_padded_both_sides(self):
        assert pad_and_merge([(5000, 6000, "a")]) == [(3500, 7500, "a")]

    def test_start_clamped_at_zero(self):
        assert pad_and_merge([(500, 1000, "a")]) == [(0, 2500, "a")]

    def test_overlapping_events_merge_and_join_labels(self):
        spans = pad_and_merge([(0, 2000, "a"), (3000, 5000, "b")])
        # padded: (0, 3500) and (1500, 6500) overlap
        assert spans == [(0, 6500, "a + b")]

    def test_touching_events_merge(self):
        # padded: (0, 3500) and (3500, 7500) touch exactly
        spans = pad_and_merge([(1000, 2000, "a"), (5000, 6000, "b")])
        assert spans == [(0, 7500, "a + b")]

    def test_disjoint_events_stay_separate(self):
        spans = pad_and_merge([(1000, 2000, "a"), (10000, 11000, "b")])
        assert spans == [(0, 3500, "a"), (8500, 12500, "b")]

    def test_unsorted_input_is_sorted(self):
        spans = pad_and_merge([(10000, 11000, "b"), (1000, 2000, "a")])
        assert [s[2] for s in spans] == ["a", "b"]

    def test_duplicate_labels_not_repeated(self):
        spans = pad_and_merge([(0, 1000, "walk"), (1500, 2500, "walk")])
        assert spans == [(0, 4000, "walk")]

    def test_custom_pad(self):
        assert pad_and_merge([(5000, 6000, "a")], pad_ms=0) == [(5000, 6000, "a")]

    def test_inverted_span_normalised(self):
        assert pad_and_merge([(6000, 5000, "a")]) == [(3500, 7500, "a")]

    def test_three_way_merge(self):
        spans = pad_and_merge(
            [(0, 1000, "a"), (2000, 3000, "b"), (4000, 5000, "c")]
        )
        assert spans == [(0, 6500, "a + b + c")]


# ─── fixture run ──────────────────────────────────────────────────────────────

def make_clip_run(tmp_path: Path) -> Path:
    """Run dir with analysis.json events and DB ethogram labels."""
    run_dir = tmp_path / "cliprun"
    run_dir.mkdir()
    db = TempoGraphDB(run_dir / "tempograph.db")
    # frames every second for 12 s
    for i in range(13):
        db.insert_frame(i * 30, i * 1000, f"frames/frame_{i*30:06d}.jpg",
                        False, 0.0)
    # ethogram: 'grooming' on frames at t=10..12s
    for i in (10, 11, 12):
        db.insert_ethogram_label(i * 30, "grooming", 0.9)
    db.close()

    analysis = {
        "entities": [
            {"id": "dog_1", "type": "dog", "first_seen": "00:02",
             "last_seen": "00:09", "description": "a dog"},
            {"id": "person_1", "type": "person", "first_seen": "00:02",
             "last_seen": "00:09", "description": "a person"},
        ],
        "visual_events": [
            {"type": "approach", "start_time": "00:02", "end_time": "00:03",
             "description": "dog approaches", "entities": ["dog_1"],
             "confidence": 0.8},
            {"type": "interact", "start_time": "00:08", "end_time": "00:09",
             "description": "dog interacts with person",
             "entities": ["dog_1", "person_1"], "confidence": 0.9},
        ],
        "audio_events": [],
        "summary": "",
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis))
    return run_dir


class TestSelectEvents:
    def test_all_events_padded_and_merged(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir / "tempograph.db")
        # approach 2-3s -> 0.5-4.5; interact 8-9s -> 6.5-10.5;
        # grooming 10-12s -> 8.5-13.5 merges with interact -> 6.5-13.5
        assert spans == [
            (500, 4500, "approach: dog_1"),
            (6500, 13500, "interact: dog_1, person_1 + grooming"),
        ]

    def test_accepts_run_dir_too(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        assert select_events(run_dir) == select_events(run_dir / "tempograph.db")

    def test_entity_filter(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir, entity="person_1")
        assert len(spans) == 1
        assert spans[0][2].startswith("interact")
        # entity filter excludes non-entity-scoped ethogram labels
        assert "grooming" not in spans[0][2]

    def test_behavior_filter_visual(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir, behavior="approach")
        assert spans == [(500, 4500, "approach: dog_1")]

    def test_behavior_filter_ethogram(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir, behavior="grooming")
        assert spans == [(8500, 13500, "grooming")]

    def test_time_range_filter(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir, time_range=(0, 5000))
        assert spans == [(500, 4500, "approach: dog_1")]

    def test_no_match_returns_empty(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        assert select_events(run_dir, entity="cat_9") == []
        assert select_events(run_dir, behavior="flying") == []

    def test_missing_analysis_uses_ethogram_only(self, tmp_path):
        run_dir = make_clip_run(tmp_path)
        (run_dir / "analysis.json").unlink()
        spans = select_events(run_dir)
        assert spans == [(8500, 13500, "grooming")]


# ─── ffmpeg integration ───────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
class TestExportClips:
    @pytest.fixture(scope="class")
    def source_video(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("clipsrc")
        video = tmp / "source14.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=14:size=320x180:rate=30",
             "-pix_fmt", "yuv420p", str(video)],
            check=True,
        )
        return video

    def test_each_span_yields_a_clip_with_matching_duration(
        self, tmp_path, source_video
    ):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir)
        assert len(spans) == 2

        result = export_clips(source_video, spans, tmp_path / "clips")
        assert len(result["clips"]) == len(spans)
        assert result["montage"] is None

        for path, (start_ms, end_ms, _label) in zip(result["clips"], spans):
            assert path.exists() and path.stat().st_size > 0
            want_s = (min(end_ms, 14000) - start_ms) / 1000.0
            got_s = _ffprobe_duration(path)
            assert got_s is not None
            assert abs(got_s - want_s) <= 0.5, (
                f"{path.name}: got {got_s:.2f}s, wanted {want_s:.2f}s"
            )

    def test_montage_is_ffprobe_valid(self, tmp_path, source_video):
        run_dir = make_clip_run(tmp_path)
        spans = select_events(run_dir)
        result = export_clips(source_video, spans, tmp_path / "clips",
                              montage=True)
        montage = result["montage"]
        assert montage is not None and montage.exists()
        dur = _ffprobe_duration(montage)
        assert dur is not None and dur > 0
        # sum of clips minus one crossfade, roughly
        clip_total = sum(_ffprobe_duration(c) for c in result["clips"])
        assert dur == pytest.approx(clip_total - 0.25, abs=1.0)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(montage)],
            capture_output=True, text=True,
        )
        assert probe.stdout.strip() == "h264"

    def test_span_beyond_source_is_clamped(self, tmp_path, source_video):
        result = export_clips(
            source_video, [(12000, 20000, "tail")], tmp_path / "clips2"
        )
        assert len(result["clips"]) == 1
        got = _ffprobe_duration(result["clips"][0])
        assert got is not None
        assert abs(got - 2.0) <= 0.5  # clamped to the 14 s source

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_clips(tmp_path / "nope.mp4", [(0, 1000, "x")],
                         tmp_path / "clips3")
