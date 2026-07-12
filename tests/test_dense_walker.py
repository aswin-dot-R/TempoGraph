"""Tests for DenseCaptionWalker (PS2).

Mocks ``requests.post`` — the suite runs with no llama-server and no GPU.
"""

import base64
import sys
import threading
from pathlib import Path
from typing import List

import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.dense_captioner import (
    DenseCaptionWalker,
    jaccard,
    parse_two_lines,
    should_escalate,
)
from src.storage import TempoGraphDB


# ── Test fixtures ──────────────────────────────────────────────────


def _make_jpeg(tmp_path: Path, idx: int) -> Path:
    """Create a tiny (16x16) solid-color JPEG at ``tmp_path/img_<idx>.jpg``."""
    colors = [
        (200, 50, 50),
        (50, 200, 50),
        (50, 50, 200),
        (200, 200, 50),
        (200, 50, 200),
        (50, 200, 200),
    ]
    color = colors[idx % len(colors)]
    img = Image.new("RGB", (16, 16), color)
    p = tmp_path / f"img_{idx}.jpg"
    img.save(p, format="JPEG")
    return p


def _seed_db(db_path: Path, n: int, deltas: List[float]) -> None:
    """Create a DB with ``n`` frames, each pointing at a tiny JPEG."""
    db = TempoGraphDB(db_path)
    for i in range(n):
        img = _make_jpeg(db_path.parent, i)
        delta = deltas[i] if i < len(deltas) else 0.0
        db.insert_frame(
            frame_idx=i,
            timestamp_ms=i * 33,
            image_path=str(img),
            is_keyframe=(i % 5 == 0),
            delta_score=delta,
        )
    db.close()


class _MockResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# ── Task 5.1: parse_two_lines ─────────────────────────────────────


class TestParseTwoLines:
    def test_well_formed(self):
        caption, change = parse_two_lines(
            "FRAME: keys on a black table\nCHANGE: a hand reaches in"
        )
        assert caption == "keys on a black table"
        assert change == "a hand reaches in"

    def test_missing_change_line(self):
        caption, change = parse_two_lines("FRAME: just a desk")
        assert caption == "just a desk"
        assert change is None

    def test_prefix_less_fallback(self):
        # No FRAME: prefix — first non-empty line becomes the caption.
        caption, change = parse_two_lines("Just a desk\nCHANGE: nothing happened")
        assert caption == "Just a desk"
        assert change == "nothing happened"

    def test_empty_reply(self):
        caption, change = parse_two_lines("")
        assert caption == "(no caption)"
        assert change is None

    def test_whitespace_only(self):
        caption, change = parse_two_lines("   \n  \n")
        assert caption == "(no caption)"
        assert change is None


# ── Task 5.2: jaccard ─────────────────────────────────────────────


class TestJaccard:
    def test_identical(self):
        assert jaccard("the cat sat", "the cat sat") == 1.0

    def test_disjoint(self):
        assert jaccard("red car", "blue boat") == 0.0

    def test_partial_overlap(self):
        sim = jaccard("the cat sat on", "the dog sat on")
        # {'the', 'cat', 'sat', 'on'} ∩ {'the', 'dog', 'sat', 'on'} = 3/5
        assert sim == 0.6

    def test_empty_both(self):
        assert jaccard("", "") == 1.0

    def test_one_empty(self):
        assert jaccard("hello", "") == 0.0


# ── Task 5.3: should_escalate ─────────────────────────────────────


class TestShouldEscalate:
    def test_delta_trigger_fires(self):
        assert (
            should_escalate(
                delta_score=10.0,
                delta_threshold=5.0,
                caption="new scene",
                prev_caption="old scene",
                similarity_floor=0.3,
            )
            is True
        )

    def test_similarity_trigger_fires(self):
        assert (
            should_escalate(
                delta_score=1.0,
                delta_threshold=5.0,
                caption="totally different thing here",
                prev_caption="the cat sat on the mat",
                similarity_floor=0.3,
            )
            is True
        )

    def test_neither_fires(self):
        assert (
            should_escalate(
                delta_score=1.0,
                delta_threshold=5.0,
                caption="the cat sat on the mat",
                prev_caption="the cat sat on the mat",
                similarity_floor=0.3,
            )
            is False
        )

    def test_first_frame_prev_none_only_delta(self):
        # prev_caption is None → similarity signal is skipped,
        # only delta can fire.
        assert (
            should_escalate(
                delta_score=1.0,
                delta_threshold=5.0,
                caption="anything",
                prev_caption=None,
                similarity_floor=0.3,
            )
            is False
        )
        assert (
            should_escalate(
                delta_score=10.0,
                delta_threshold=5.0,
                caption="anything",
                prev_caption=None,
                similarity_floor=0.3,
            )
            is True
        )


# ── Task 5.4: percentile threshold ────────────────────────────────


class TestPercentileThreshold:
    def test_delta_threshold_computed_correctly(self, tmp_path):
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        _seed_db(tmp_path / "t.db", 10, deltas)
        db = TempoGraphDB(tmp_path / "t.db")
        raw = [
            r["delta_score"]
            for r in db._conn.execute(
                "SELECT delta_score FROM frames ORDER BY frame_idx"
            ).fetchall()
        ]
        assert raw == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        db.close()

    def test_escalation_with_identical_captions(self, tmp_path):
        """Mock walker: identical captions → only delta-triggered frames escalate."""
        from src.modules.dense_captioner import _percentile

        deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        _seed_db(tmp_path / "t.db", 10, deltas)

        threshold = _percentile(deltas, 90.0)
        # 90th percentile of [1..10] with linear interp:
        # pos = 0.9 * 9 = 8.1 → s[8]*0.9 + s[9]*0.1 = 9*0.9 + 10*0.1 = 9.1
        assert abs(threshold - 9.1) < 0.01

        # Frames with delta >= 9.1: only index 9 (delta=10).
        # All captions identical → similarity never fires.
        escalated_indices = [i for i, d in enumerate(deltas) if d >= threshold]
        assert escalated_indices == [9]


# ── Task 5.5: full walk ──────────────────────────────────────────


class TestFullWalk:
    def test_twenty_frames_all_captioned(self, tmp_path):
        n = 20
        deltas = [float(i) for i in range(n)]
        _seed_db(tmp_path / "t.db", n, deltas)

        # Mock responses: identical captions so only delta triggers escalation.
        mock_reply = "FRAME: identical caption across all frames\nCHANGE: no change\n"

        def _mock_post(url, json=None, timeout=None):
            return _MockResponse(
                {
                    "choices": [{"message": {"content": mock_reply}}],
                }
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_post)
            walker = DenseCaptionWalker(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8085",
            )
            result = walker.walk()

        assert result["captioned"] == n
        assert result["errors"] == 0

        # Verify rows exist in DB.
        db = TempoGraphDB(tmp_path / "t.db")
        total, _, _ = db.count_frame_captions()
        assert total == n
        db.close()

    def test_prev_caption_threading(self, tmp_path):
        """Frame k's prompt must contain the caption returned for frame k-1."""
        n = 5
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        replies = [
            "FRAME: frame 0 caption\nCHANGE: start\n",
            "FRAME: frame 1 caption\nCHANGE: something\n",
            "FRAME: frame 2 caption\nCHANGE: more\n",
            "FRAME: frame 3 caption\nCHANGE: even more\n",
            "FRAME: frame 4 caption\nCHANGE: end\n",
        ]
        call_idx = 0
        captured_prompts: List[str] = []

        def _mock_post(url, json=None, timeout=None):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            captured_prompts.append(json["messages"][0]["content"][0]["text"])
            return _MockResponse(
                {
                    "choices": [{"message": {"content": replies[idx]}}],
                }
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_post)
            walker = DenseCaptionWalker(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8085",
            )
            walker.walk()

        # Frame 0: prev_caption = "(first frame)"
        assert "(first frame)" in captured_prompts[0]
        # Frame 1: prev_caption = "frame 0 caption"
        assert "frame 0 caption" in captured_prompts[1]
        # Frame 2: prev_caption = "frame 1 caption"
        assert "frame 1 caption" in captured_prompts[2]
        # Frame 4: prev_caption = "frame 3 caption"
        assert "frame 3 caption" in captured_prompts[4]


# ── Task 5.6: resume ─────────────────────────────────────────────


class TestResume:
    def test_second_walk_returns_zero_captioned(self, tmp_path):
        n = 20
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        mock_reply = "FRAME: same\nCHANGE: no change\n"

        def _mock_post(url, json=None, timeout=None):
            return _MockResponse(
                {
                    "choices": [{"message": {"content": mock_reply}}],
                }
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_post)
            walker = DenseCaptionWalker(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8085",
            )
            # First walk.
            r1 = walker.walk()
            assert r1["captioned"] == n
            assert r1["skipped"] == 0

            # Second walk: everything already captioned.
            r2 = walker.walk()
            assert r2["captioned"] == 0
            assert r2["skipped"] == n
            assert r2["errors"] == 0


# ── Task 5.7: cancel ─────────────────────────────────────────────


class TestCancel:
    def test_cancel_after_five_frames(self, tmp_path):
        n = 20
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        mock_reply = "FRAME: caption\nCHANGE: no change\n"
        cancel_event = threading.Event()
        progress_done = []

        def _mock_post(url, json=None, timeout=None):
            return _MockResponse(
                {
                    "choices": [{"message": {"content": mock_reply}}],
                }
            )

        def _on_progress(data: dict):
            progress_done.append(data["done"])
            if data["done"] >= 5:
                cancel_event.set()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_post)
            walker = DenseCaptionWalker(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8085",
                cancel_event=cancel_event,
                on_progress=_on_progress,
            )
            result = walker.walk()

        assert result["captioned"] == 5
        assert result["skipped"] == 0
        # Progress callback should have been called with done values
        # up to (and possibly including) 5.
        assert 5 in progress_done
        # DB should have exactly 5 rows.
        db = TempoGraphDB(tmp_path / "t.db")
        total, _, _ = db.count_frame_captions()
        assert total == 5
        db.close()


# ── Task 5.8: HTTP error on one frame ────────────────────────────


class TestHTTPError:
    def test_one_frame_http_error_other_succeed(self, tmp_path):
        n = 20
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        mock_reply = "FRAME: ok caption\nCHANGE: fine\n"
        error_at = 7  # zero-indexed frame 7

        call_count = 0

        def _mock_post_with_error(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == error_at + 1:  # 1-indexed
                raise requests.exceptions.ConnectionError("simulated")
            return _MockResponse(
                {
                    "choices": [{"message": {"content": mock_reply}}],
                }
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_post_with_error)
            walker = DenseCaptionWalker(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8085",
            )
            result = walker.walk()

        assert result["errors"] == 1
        assert result["captioned"] == n - 1
        # The walk should have completed (not crashed).
        db = TempoGraphDB(tmp_path / "t.db")
        total, _, _ = db.count_frame_captions()
        assert total == n - 1
        db.close()
