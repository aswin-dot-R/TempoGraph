"""Tests for EscalationVerifier and run_dense_captioning (PS3).

Mocks ``requests.post`` — the suite runs with no llama-server and no GPU.
Routes by URL: ``:8085`` → walker replies, ``:8096`` → verifier replies.
"""

import sys
import threading
import time
from pathlib import Path
from typing import List

import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.dense_captioner import (
    DenseCaptionWalker,
    EscalationVerifier,
    parse_verdict,
    run_dense_captioning,
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


def _seed_escalations(db_path: Path, frame_indices: List[int]) -> None:
    """Insert escalated walker rows (no verifier verdict yet) for given frames."""
    db = TempoGraphDB(db_path)
    from datetime import datetime, timezone

    for idx in frame_indices:
        db._conn.execute(
            "INSERT OR REPLACE INTO frame_captions "
            "(frame_idx, caption, change_line, walker_model, escalated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                idx,
                f"escalated caption {idx}",
                f"big change at {idx}",
                "ornith-1.0-9b",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    db._conn.commit()
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


# ── Task 4.1: parse_verdict ───────────────────────────────────────


class TestParseVerdict:
    def test_agree(self):
        agrees, caption = parse_verdict(
            "VERDICT: AGREE — the caption is accurate.\n"
            "CAPTION: a cat sitting on a mat"
        )
        assert agrees is True
        assert caption == "a cat sitting on a mat"

    def test_disagree(self):
        agrees, caption = parse_verdict(
            "VERDICT: DISAGREE — wrong scene.\n" "CAPTION: a dog running in a park"
        )
        assert agrees is False
        assert caption == "a dog running in a park"

    def test_missing_verdict_fallback(self):
        # No VERDICT line → benefit of the doubt → agrees=True.
        agrees, caption = parse_verdict(
            "Just some text here.\n" "CAPTION: a desk with books"
        )
        assert agrees is True
        assert caption == "a desk with books"

    def test_missing_caption_fallback(self):
        # No CAPTION line → whole reply stripped.
        agrees, caption = parse_verdict("VERDICT: AGREE — looks fine")
        assert agrees is True
        assert caption == "VERDICT: AGREE — looks fine"

    def test_empty_reply(self):
        agrees, caption = parse_verdict("")
        assert agrees is True
        assert caption == "(no caption)"

    def test_disagree_contains_agree_substring(self):
        # "DISAGREE" contains "agree" as substring, but also "disagree"
        # → the condition "agree and not disagree" → False.
        agrees, _ = parse_verdict("VERDICT: DISAGREE")
        assert agrees is False

    def test_lowercase_verdict(self):
        agrees, _ = parse_verdict("verdict: agree")
        assert agrees is True


# ── Task 4.2: verifier alone on pre-populated DB ─────────────────


class TestVerifierAlone:
    def test_five_escalated_rows_all_verified(self, tmp_path):
        n = 20
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        # Pre-populate 5 escalated rows.
        escalated_indices = [2, 5, 8, 12, 17]
        _seed_escalations(tmp_path / "t.db", escalated_indices)

        # Mock verifier: alternate AGREE/DISAGREE.
        call_count = 0

        def _mock_verifier_post(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                reply = (
                    "VERDICT: AGREE\n"
                    "CAPTION: verifier caption for frame "
                    f"{call_count}\n"
                )
            else:
                reply = (
                    "VERDICT: DISAGREE — not a match.\n"
                    "CAPTION: alternative caption "
                    f"{call_count}\n"
                )
            return _MockResponse({"choices": [{"message": {"content": reply}}]})

        # walker_done is pre-set so the verifier drains and exits.
        walker_done = threading.Event()
        walker_done.set()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_verifier_post)
            verifier = EscalationVerifier(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8096",
                walker_done=walker_done,
            )
            result = verifier.run()

        assert result["verified"] == 5
        assert result["agreed"] == 3  # calls 1,3,5 are AGREE
        assert result["disagreed"] == 2  # calls 2,4 are DISAGREE
        assert result["errors"] == 0

        # Verify DB has verdicts.
        db = TempoGraphDB(tmp_path / "t.db")
        total, escalated, verified = db.count_frame_captions()
        assert total == 5
        assert escalated == 5
        assert verified == 5
        # Check verifier_model is set.
        for idx in escalated_indices:
            row = db.get_frame_caption(idx)
            assert row is not None
            assert row["verifier_model"] == "ornith-1.0-35b"
            assert row["verifier_caption"] is not None
            assert row["verified_at"] is not None
        db.close()


# ── Task 4.3: retry cap ──────────────────────────────────────────


class TestRetryCap:
    def test_retry_cap_skips_row_after_three_failures(self, tmp_path):
        n = 20
        deltas = [0.0] * n
        _seed_db(tmp_path / "t.db", n, deltas)

        # Pre-populate 3 escalated rows; ALL will fail.
        _seed_escalations(tmp_path / "t.db", [3, 7, 11])

        call_count = 0

        def _mock_always_fail(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.ConnectionError("simulated server down")

        walker_done = threading.Event()
        walker_done.set()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("requests.post", _mock_always_fail)
            verifier = EscalationVerifier(
                db_path=tmp_path / "t.db",
                base_url="http://127.0.0.1:8096",
                walker_done=walker_done,
            )
            result = verifier.run()

        # All 3 frames fail 3 times each, then get skipped.
        assert result["verified"] == 0
        assert result["errors"] == 9  # 3 frames * 3 retries
        assert call_count == 9

        # All frames should still be unverified.
        db = TempoGraphDB(tmp_path / "t.db")
        for idx in [3, 7, 11]:
            row = db.get_frame_caption(idx)
            assert row is not None
            assert row["verifier_agrees"] is None
            assert row["verifier_caption"] is None
        db.close()


# ── Task 4.4: true parallel run ─────────────────────────────────


class TestParallelRun:
    def test_parallel_run_with_overlap_proof(self, tmp_path):
        n = 20
        # Use deltas that ensure exactly 6 frames escalate via delta trigger.
        # With [0]*14 + [100]*6, 90th percentile ≈ 10.0, so frames 14-19 escalate.
        deltas = [0.0] * 14 + [100.0] * 6
        _seed_db(tmp_path / "t.db", n, deltas)

        # Walker mock: caption each frame, escalate frames 14,15,16,17,18,19.
        escalated_frames = {14, 15, 16, 17, 18, 19}
        walker_call_count = 0
        walker_timeline = []

        def _mock_walker_post(url, json=None, timeout=None):
            nonlocal walker_call_count
            idx = walker_call_count
            walker_call_count += 1
            walker_timeline.append(time.time())
            # Small delay between frames to give verifier time to poll.
            time.sleep(0.02)
            caption = f"caption for frame {idx}"
            if idx in escalated_frames:
                change = f"big change at frame {idx}"
            else:
                change = "no change"
            return _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": f"FRAME: {caption}\nCHANGE: {change}\n"
                            }
                        }
                    ]
                }
            )

        # Verifier mock: all AGREE with captions.
        verifier_call_count = 0

        def _mock_verifier_post(url, json=None, timeout=None):
            nonlocal verifier_call_count
            verifier_call_count += 1
            return _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "VERDICT: AGREE\n"
                                "CAPTION: verifier says this is fine\n"
                            }
                        }
                    ]
                }
            )

        # Collect progress events to prove overlap.
        progress_events = []
        verifier_event_times = []

        def _on_progress(data: dict):
            progress_events.append(data)
            if data.get("who") == "verifier":
                verifier_event_times.append(data.get("time", time.time()))

        cancel_event = threading.Event()

        with pytest.MonkeyPatch.context() as mp:
            # Route by URL.
            def _routed_post(url, json=None, timeout=None):
                if "8085" in url:
                    return _mock_walker_post(url, json, timeout)
                elif "8096" in url:
                    return _mock_verifier_post(url, json, timeout)
                raise AssertionError(f"Unexpected URL: {url}")

            mp.setattr("requests.post", _routed_post)
            result = run_dense_captioning(
                db_path=tmp_path / "t.db",
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                on_progress=_on_progress,
                cancel_event=cancel_event,
                poll_interval_s=0.05,  # fast polling for overlap proof
            )

        # Walker captioned all 20 frames, 6 escalated.
        assert result["walker"]["captioned"] == n
        assert result["walker"]["escalated"] == len(escalated_frames), (
            f"Expected {len(escalated_frames)} escalated, got "
            f"{result['walker']['escalated']}"
        )
        assert result["walker"]["errors"] == 0

        # Verifier verified all 6 escalated rows.
        assert (
            result["verifier"]["verified"] == 6
        ), f"Expected 6 verified, got {result['verifier']['verified']}"
        assert result["verifier"]["agreed"] == 6
        assert result["verifier"]["disagreed"] == 0

        # Every escalated row has a verdict.
        db = TempoGraphDB(tmp_path / "t.db")
        for idx in escalated_frames:
            row = db.get_frame_caption(idx)
            assert row is not None, f"Frame {idx} has no caption"
            assert row["verifier_agrees"] is not None, f"Frame {idx} has no verdict"
            assert row["verified_at"] is not None
        db.close()

        # No "database is locked" — if there were locking errors,
        # the verifier would have had errors > 0 or the run would have
        # stalled. We check errors == 0.
        assert result["verifier"]["errors"] == 0

        # Overlap proof: at least one verifier event happened before
        # the walker finished.
        assert len(verifier_event_times) > 0
        walker_finish_time = walker_timeline[-1] if walker_timeline else 0
        any_before = any(t < walker_finish_time for t in verifier_event_times)
        assert any_before, (
            "No verifier verdict was written before walker finished — "
            "no overlap detected."
        )


# ── Task 4.5: cancel ─────────────────────────────────────────────


class TestCancel:
    def test_cancel_stops_both_threads_cleanly(self, tmp_path):
        n = 20
        # Use a mix of deltas so some frames escalate, giving the verifier
        # something to do between walker frames.
        deltas = [0.0] * 10 + [50.0] * 10
        _seed_db(tmp_path / "t.db", n, deltas)

        walker_call_count = 0
        cancel_event = threading.Event()
        progress_calls = []

        def _mock_walker_post(url, json=None, timeout=None):
            nonlocal walker_call_count
            walker_call_count += 1
            idx = walker_call_count - 1
            if idx >= 10:
                change = "big change"
            else:
                change = "no change"
            return _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": f"FRAME: caption {idx}\nCHANGE: {change}\n"
                            }
                        }
                    ]
                }
            )

        verifier_call_count = 0

        def _mock_verifier_post(url, json=None, timeout=None):
            nonlocal verifier_call_count
            verifier_call_count += 1
            return _MockResponse(
                {"choices": [{"message": {"content": "VERDICT: AGREE\nCAPTION: ok\n"}}]}
            )

        def _on_progress(data: dict):
            progress_calls.append(data)
            # Cancel after walker has processed ~5 frames.
            if data.get("who") == "walker":
                done = data.get("done", 0)
                if done >= 5 and not cancel_event.is_set():
                    cancel_event.set()

        with pytest.MonkeyPatch.context() as mp:

            def _routed_post(url, json=None, timeout=None):
                if "8085" in url:
                    return _mock_walker_post(url, json, timeout)
                elif "8096" in url:
                    return _mock_verifier_post(url, json, timeout)
                raise AssertionError(f"Unexpected URL: {url}")

            mp.setattr("requests.post", _routed_post)

            # Pass verifier-specific kwargs separately.
            verifier_kwargs = {"poll_interval_s": 0.1}
            result = run_dense_captioning(
                db_path=tmp_path / "t.db",
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                on_progress=_on_progress,
                cancel_event=cancel_event,
                **verifier_kwargs,
            )

        # Both threads stopped cleanly.
        assert result["walker"]["captioned"] < n  # partial
        # Verifier should have stopped too (partial counts).
        assert isinstance(result["verifier"]["verified"], int)
        # The cancel_event should have been set.
        assert (
            cancel_event.is_set()
        ), f"Cancel not triggered. Progress calls: {progress_calls}"
