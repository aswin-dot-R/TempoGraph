"""Tests for summarizer integration in the Results Overview tab.

Verifies:
- generate_summary() with a fake LLM callable returns the fake's text.
- generate_summary() with no LLM callable produces a non-empty heuristic summary.
- Summary caching: second call with cached result does not invoke the LLM callable.
"""

import sqlite3
from unittest.mock import patch

from src.summarizer import generate_summary


# ── fixture data ────────────────────────────────────────────────────────

_FIXTURE_ENTITIES = [
    {
        "id": "person_1",
        "type": "person",
        "description": "A person in a red jacket walking near the entrance",
        "first_seen": "00:05",
        "last_seen": "01:30",
    },
    {
        "id": "car_1",
        "type": "car",
        "description": "A parked silver car in the lot",
        "first_seen": "00:00",
        "last_seen": "02:00",
    },
]

_FIXTURE_EVENTS = [
    {
        "type": "approach",
        "entities": ["person_1"],
        "start_time": "00:10",
        "end_time": "00:20",
        "description": "Person approaches the car",
        "confidence": 0.85,
    },
]

_FIXTURE_AUDIO = [
    {
        "type": "speech",
        "start_time": "00:15",
        "end_time": "00:18",
        "text": "Hey, watch out!",
    },
]


# ── tests ───────────────────────────────────────────────────────────────


class TestGenerateSummaryWithLLM:
    """With an injectable LLM callable, the summary is whatever the callable returns."""

    def test_fake_callable_returns_verbatim(self):

        fake_result = "This is the fake summary."
        result = generate_summary(
            entities=_FIXTURE_ENTITIES,
            visual_events=_FIXTURE_EVENTS,
            audio_events=_FIXTURE_AUDIO,
            summary_text="Raw summary text",
            llm_callable=lambda prompt: fake_result,
        )
        assert result == fake_result

    def test_fake_callable_receives_prompt(self):
        """The callable should receive a formatted prompt."""
        captured_prompt = []

        def capturing_callable(prompt):
            captured_prompt.append(prompt)
            return "ok"

        generate_summary(
            entities=_FIXTURE_ENTITIES,
            visual_events=_FIXTURE_EVENTS,
            audio_events=_FIXTURE_AUDIO,
            summary_text="Some summary",
            llm_callable=capturing_callable,
        )

        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]
        assert "person_1" in prompt
        assert "approach" in prompt
        assert "Hey, watch out!" in prompt
        assert "Hey, watch out!" in prompt


class TestGenerateSummaryHeuristic:
    """Without an LLM callable, generate_summary uses heuristic fallback."""

    def test_heuristic_non_empty(self):
        result = generate_summary(
            entities=_FIXTURE_ENTITIES,
            visual_events=_FIXTURE_EVENTS,
            audio_events=_FIXTURE_AUDIO,
            summary_text="Raw summary text",
        )
        assert len(result) > 0

    def test_heuristic_contains_entity_name(self):
        result = generate_summary(
            entities=_FIXTURE_ENTITIES,
            visual_events=_FIXTURE_EVENTS,
            audio_events=[],
            summary_text="Raw summary text",
        )
        assert "person" in result.lower() or "car" in result.lower()

    def test_heuristic_empty_entities(self):
        result = generate_summary(
            entities=[],
            visual_events=[],
            audio_events=[],
            summary_text="",
        )
        assert len(result) > 0
        assert "Video analysis completed." in result

    def test_heuristic_audio_speech(self):
        result = generate_summary(
            entities=[],
            visual_events=[],
            audio_events=[{"type": "speech", "text": "Hello"}],
            summary_text="",
        )
        assert "speech" in result.lower() or "1 speech" in result.lower()


class TestSummaryCaching:
    """Summary caching: second call with cached result does not invoke LLM callable."""

    def test_cached_summary_skips_llm(self, tmp_path):
        """When a cached summary exists in run_meta, _generate_run_summary
        should return it without calling the LLM backend.

        This tests the caching flow by patching _llm_health_probe to
        return True (LLM reachable) but verifying the LLM callable is not invoked.
        """
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        db_path = run_dir / "tempograph.db"

        # Create a DB with all required tables and a pre-cached summary
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS frames ("
            "frame_idx INTEGER PRIMARY KEY, timestamp_ms INTEGER NOT NULL, "
            "image_path TEXT NOT NULL, is_keyframe INTEGER NOT NULL, "
            "delta_score REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO frames (frame_idx, timestamp_ms, image_path, "
            "is_keyframe, delta_score) VALUES (1, 0, '/tmp/1.jpg', 0, 0.0)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            ("summary", "CACHED_SUMMARY_TEXT"),
        )
        conn.commit()
        conn.close()

        analysis = {
            "entities": _FIXTURE_ENTITIES,
            "visual_events": _FIXTURE_EVENTS,
            "audio_events": _FIXTURE_AUDIO,
            "summary": "raw text",
        }

        from ui.pages.Results import _generate_run_summary

        # Re-open the connection for _generate_run_summary
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Patch the LLM probe to always return True (LLM is reachable)
        # The cached summary should be returned without calling the LLM
        with patch("ui.pages.Results._llm_health_probe", return_value=True):
            with patch(
                "ui.pages.Results._llm_call",
                side_effect=RuntimeError("Should not be called!"),
            ) as mock_call:
                result = _generate_run_summary(analysis, conn, run_dir)

        conn.close()

        # The cached summary should be returned
        assert result == "CACHED_SUMMARY_TEXT"
        # The LLM callable should NOT have been invoked
        mock_call.assert_not_called()


class TestLlmHealthProbe:
    """The health probe should return False for unreachable URLs."""

    def test_unreachable_url_returns_false(self):
        from ui.pages.Results import _llm_health_probe

        assert _llm_health_probe("http://127.0.0.1:1") is False
