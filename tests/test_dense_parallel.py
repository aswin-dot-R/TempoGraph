"""Tests for DenseCaptionWalker — two-phase parallel walker.

Parity, prompt, ordering, phase-2 failure tolerance, resume, cancel,
verifier transcript, and get_audio_segments_overlapping.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import sqlite3

# ── helpers ──────────────────────────────────────────────────────────


def _create_test_db(db_path: str, rows=None) -> Path:
    """Create a test database with frames and frame_captions tables.

    Args:
        db_path: Path to the database file.
        rows: List of dicts with frame data.

    Returns:
        The db_path.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("DROP TABLE IF EXISTS frames")
    conn.execute("DROP TABLE IF EXISTS frame_captions")
    conn.execute("DROP TABLE IF EXISTS audio_segments")

    conn.execute("""
        CREATE TABLE frames (
            frame_idx INTEGER PRIMARY KEY,
            timestamp_ms INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            is_keyframe INTEGER NOT NULL,
            delta_score REAL NOT NULL
        )
        """)
    conn.execute("""
        CREATE TABLE frame_captions (
            frame_idx INTEGER PRIMARY KEY,
            caption TEXT NOT NULL,
            change_line TEXT,
            walker_model TEXT NOT NULL,
            escalated INTEGER NOT NULL DEFAULT 0,
            verifier_caption TEXT,
            verifier_agrees INTEGER,
            verifier_model TEXT,
            created_at TEXT NOT NULL,
            verified_at TEXT,
            prompt TEXT
        )
        """)
    conn.execute("""
        CREATE TABLE audio_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """)

    if rows:
        for row in rows:
            conn.execute(
                "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?)",
                (
                    row["frame_idx"],
                    row.get("timestamp_ms", 0),
                    row["image_path"],
                    row.get("is_keyframe", 0),
                    row["delta_score"],
                ),
            )
            if "caption" in row:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO frame_captions
                    (frame_idx, caption, change_line, walker_model, escalated,
                     created_at, prompt)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        row["frame_idx"],
                        row["caption"],
                        row.get("change_line", ""),
                        row.get("walker_model", ""),
                        row.get("escalated", 0),
                        "",
                        row.get("prompt", ""),
                    ),
                )

    conn.commit()
    conn.close()
    return db_path


# ── Tests ───────────────────────────────────────────────────────────


class TestDenseCaptionWalkerParity:
    """Parity: prompt format, model, max_tokens, temperature."""

    def test_prompt_format(self):
        """``DenseCaptionWalker`` prompt template must contain ``Previous frame:``."""
        from src.modules.dense_captioner import _FRAME_PROMPT

        assert "Previous frame:" in _FRAME_PROMPT

    def test_parallel_prompt_no_previous_frame(self):
        """Parallel-safe prompt must NOT contain ``Previous frame:``."""
        from src.modules.dense_captioner import _PARALLEL_FRAME_PROMPT

        assert "Previous frame:" not in _PARALLEL_FRAME_PROMPT

    def test_model_name(self):
        """``DenseCaptionWalker.model_name`` must equal the model kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.model_name == "test-9b"

    def test_max_tokens(self):
        """``DenseCaptionWalker.max_tokens`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=256,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.max_tokens == 256

    def test_temperature(self):
        """``DenseCaptionWalker.temperature`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.15,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.temperature == 0.15

    def test_concurrency(self):
        """``DenseCaptionWalker.concurrency`` must equal the kwarg
        after ``_resolve_concurrency`` is called."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=8,
        )
        assert walker._resolve_concurrency() == 8

    def test_request_timeout(self):
        """``DenseCaptionWalker.request_timeout_s`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=60.0,
            concurrency=1,
        )
        assert walker.request_timeout_s == 60.0

    def test_base_url(self):
        """``DenseCaptionWalker.base_url`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:9999",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.base_url == "http://127.0.0.1:9999"


class TestDenseCaptionWalkerWalk:
    """Walk tests: ordering, phase-2 failure tolerance, resume, cancel."""

    def test_ordering(self):
        """Frames must be visited in ascending ``frame_idx`` order."""
        tmp_path = Path("/tmp/tg_test_ordering")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 5,
                    "image_path": "/tmp/frame_005.jpg",
                    "delta_score": 0.3,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.15,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            with patch("requests.post") as mock_post:

                def side_effect(url, json=None, **kwargs):
                    content = json.get("messages", [{}])[0].get("content", "")
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "choices": [
                                {
                                    "message": {
                                        "content": "FRAME: a cat on a mat\nCHANGE: cat appeared"
                                    }
                                }
                            ]
                        },
                    )

                mock_post.side_effect = side_effect

                result = walker.walk()
                assert result["captioned"] == 4
                assert result["errors"] == 0

    def test_phase2_http_failure_tolerance(self):
        """If phase-2 HTTP fails, escalation should still be computed from delta."""
        tmp_path = Path("/tmp/tg_test_phase2")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        # Create walker with concurrency=2 (triggers parallel path)
        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=2,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            # Mock phase-1 HTTP (caption requests)
            phase1_responses = [
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [{"message": {"content": "FRAME: a cat on a mat"}}]
                    },
                ),
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [{"message": {"content": "FRAME: a dog on a bed"}}]
                    },
                ),
            ]

            # Mock phase-2 HTTP (change-line requests) — raise ConnectionError for second call
            phase2_failures = [
                ConnectionError("Connection failed"),
                ConnectionError("Connection failed"),
            ]

            call_count = [0]

            def side_effect(url, json=None, **kwargs):
                call_count[0] += 1
                if call_count[0] <= len(phase1_responses):
                    return phase1_responses[call_count[0] - 1]
                else:
                    raise phase2_failures[call_count[0] - len(phase1_responses)]

            with patch("requests.post", side_effect=side_effect):
                result = walker.walk()
                assert result["captioned"] > 0

    def test_cancel_event(self):
        """``DenseCaptionWalker.walk`` should respect cancel event."""
        tmp_path = Path("/tmp/tg_test_cancel")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        # Create walker with cancel event
        cancel_event = threading.Event()
        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
            cancel_event=cancel_event,
        )

        # Set cancel event before walk
        cancel_event.set()

        # Walk should complete quickly (nothing to do)
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            result = walker.walk()
            assert result is not None

    def test_resume(self):
        """Pre-insert captions for half the frames → phase 1 only requests missing half."""
        tmp_path = Path("/tmp/tg_test_resume")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
                {
                    "frame_idx": 4,
                    "image_path": "/tmp/frame_004.jpg",
                    "delta_score": 0.4,
                },
            ],
        )

        # Pre-insert captions for frames 1 and 2
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at, prompt) VALUES (?,?,?,?,?,?,?)",
            (1, "pre-capt1", "", "test-9b", 0, "", ""),
        )
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at, prompt) VALUES (?,?,?,?,?,?,?)",
            (2, "pre-capt2", "", "test-9b", 0, "", ""),
        )
        conn.commit()
        conn.close()

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            call_count = [0]

            def side_effect(url, json=None, **kwargs):
                call_count[0] += 1
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [
                            {"message": {"content": f"FRAME: frame {call_count[0]}"}}
                        ]
                    },
                )

            with patch("requests.post", side_effect=side_effect):
                result = walker.walk()
                # Should only request frames 3 and 4 (2 frames)
                assert result["captioned"] == 2
                assert result["skipped"] == 2

    def test_verifyer_transcript(self, tmp_path):
        """Audio segments in the DB → the verifier prompt contains 'Spoken audio'.
        No segments → prompt doesn't. Overlap boundary: a segment ending exactly
        at ts - 5000 is excluded, one straddling the edge is included."""
        from src.storage import TempoGraphDB

        db_path = str(tmp_path / "test.db")

        # Create database with frames and audio segments
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE frames (
                frame_idx INTEGER PRIMARY KEY,
                timestamp_ms INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                is_keyframe INTEGER NOT NULL,
                delta_score REAL NOT NULL
            )
            """)
        conn.execute("""
            CREATE TABLE frame_captions (
                frame_idx INTEGER PRIMARY KEY,
                caption TEXT NOT NULL,
                change_line TEXT,
                walker_model TEXT NOT NULL,
                escalated INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                prompt TEXT
            )
            """)
        conn.execute("""
            CREATE TABLE audio_segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """)

        # Insert a frame at timestamp 10000
        conn.execute("INSERT INTO frames VALUES (1, 10000, '/tmp/frame.jpg', 0, 0.1)")

        # Insert a captioned row (escalated so the verifier picks it up)
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at) VALUES (?,?,?,?,?,?)",
            (1, "test caption", "change", "test-9b", 1, ""),
        )

        # Audio segments that overlap with the query range [5000, 15000]
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (?,?,?)",
            (4000, 6000, "Spoken audio segment 1"),
        )
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (?,?,?)",
            (14000, 16000, "Spoken audio segment 2"),
        )
        conn.commit()
        conn.close()

        # Verify get_audio_segments_overlapping returns the segments
        db = TempoGraphDB(db_path)
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)
        assert len(segments) == 2
        assert any(seg["text"] == "Spoken audio segment 1" for seg in segments)
        assert any(seg["text"] == "Spoken audio segment 2" for seg in segments)
        db.close()

        # Verify the verifier's _VERDICT_PROMPT includes {caption} and {change_line}
        from src.modules.dense_captioner import _VERDICT_PROMPT

        assert "caption" in _VERDICT_PROMPT
        assert "change_line" in _VERDICT_PROMPT

        # Verify the verifier's _VERDICT_PROMPT includes {caption} and {change_line}
        from src.modules.dense_captioner import _VERDICT_PROMPT

        assert "caption" in _VERDICT_PROMPT
        assert "change_line" in _VERDICT_PROMPT

        # Test that get_audio_segments_overlapping returns segments
        db = TempoGraphDB(db_path)
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)
        assert len(segments) == 2

        # Verify the segments include the expected text
        texts = [seg["text"] for seg in segments]
        assert "Spoken audio segment 1" in texts
        assert "Spoken audio segment 2" in texts
        db.close()

    def test_get_audio_segments_overlapping_boundary(self):
        """``get_audio_segments_overlapping`` boundary semantics:
        start-exclusive/end-exclusive.

        A segment ending exactly at ts - 5000 is excluded.
        A segment straddling the edge is included.
        """
        from src.storage import TempoGraphDB

        tmp_path = Path("/tmp/tg_test_audio_boundary")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        # Create a minimal database with frames and audio_segments
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS frames")
        conn.execute("DROP TABLE IF EXISTS audio_segments")

        conn.execute("""
            CREATE TABLE frames (
                frame_idx INTEGER PRIMARY KEY,
                timestamp_ms INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                is_keyframe INTEGER NOT NULL,
                delta_score REAL NOT NULL
            )
            """)
        conn.execute("""
            CREATE TABLE audio_segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """)

        # Insert a frame at timestamp 10000
        conn.execute("INSERT INTO frames VALUES (1, 10000, '/tmp/frame.jpg', 0, 0.1)")

        # Segment A: ends exactly at ts - 5000 (5000) — should be excluded
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (0, 5000, 'segment A')"
        )

        # Segment B: straddles the edge (4000-6000) — should be included
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (4000, 6000, 'segment B')"
        )

        # Segment C: starts after the edge (7000-9000) — should be included
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (7000, 9000, 'segment C')"
        )

        conn.commit()
        conn.close()

        db = TempoGraphDB(db_path)
        # Query range [ts-5000, ts+5000] = [5000, 15000]
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)

        # Segment A: end=5000, ts=10000 → start_ms < 15000 AND end_ms > 5000
        # 5000 > 5000 is False → excluded
        assert not any(seg["text"] == "segment A" for seg in segments)

        # Segment B: start=4000 < 15000 AND end=6000 > 5000 → included
        assert any(seg["text"] == "segment B" for seg in segments)

        # Segment C: start=7000 < 15000 AND end=9000 > 5000 → included
        assert any(seg["text"] == "segment C" for seg in segments)

        db.close()


# ── TestDynamicSlots: concurrency resolution ──────────────────────────


class TestDynamicSlots:
    """Verify the lazy _resolve_concurrency() precedence."""

    def test_probe_total_slots_50(self, tmp_path):
        """Probe returns total_slots=50 → concurrency becomes 50."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"total_slots": 50}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = walker._resolve_concurrency()

        assert result == 50
        mock_get.assert_called_once_with("http://127.0.0.1:8085/props", timeout=3.0)

    def test_probe_total_slots_200_clamped(self, tmp_path):
        """Probe returns total_slots=200 → clamped to 64."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"total_slots": 200}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = walker._resolve_concurrency()

        assert result == 64
        mock_get.assert_called_once()

    def test_probe_connection_error_falls_back(self, tmp_path):
        """Probe raises ConnectionError → falls back to settings default 4."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker
        from src.settings import get_settings

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        with patch("requests.get", side_effect=ConnectionError("refused")) as mock_get:
            result = walker._resolve_concurrency()

        assert result == get_settings().walker_concurrency
        assert result == 4
        mock_get.assert_called_once()

    def test_env_var_skips_probe(self, tmp_path, monkeypatch):
        """TEMPOGRAPH_WALKER_CONCURRENCY env var → settings default 4,
        and requests.get is NOT called."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        monkeypatch.setenv("TEMPOGRAPH_WALKER_CONCURRENCY", "7")

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        with patch("requests.get") as mock_get:
            result = walker._resolve_concurrency()

        assert result == 7
        mock_get.assert_not_called()

    def test_explicit_kwarg_skips_probe(self, tmp_path):
        """Explicit concurrency=2 → 2, and requests.get is NOT called."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=2,
        )

        with patch("requests.get") as mock_get:
            result = walker._resolve_concurrency()

        assert result == 2
        mock_get.assert_not_called()
