"""Tests for ``src/search.py`` + the Search tab, written in parallel with the
implementation (``docs/ps/ps6a.md``). Code against the FROZEN CONTRACTS in that
file — do not wait for it to land. Failing until it does is expected; the gate
is clean collection.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import TempoGraphDB


# --------------------------------------------------------------------------- #
# Top-of-file import guard (ps6b spec).
# --------------------------------------------------------------------------- #
pytest.importorskip("src.search")


# ── helpers ─────────────────────────────────────────────────────────────────


def _seed_db(tmp_path: Path, name: str = "srch") -> Path:
    """Build a small run dir: frames on disk + DB + analysis.json."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "tempograph.db"

    # 5 synthetic frames
    for i in range(5):
        img = Image.new("RGB", (64, 64), (i * 50, 100, 200))
        img.save(str(run_dir / f"f{i}.jpg"), format="JPEG")
        db = TempoGraphDB(db_path)
        db.insert_frame(
            frame_idx=i,
            timestamp_ms=i * 5000,
            image_path=str(run_dir / f"f{i}.jpg"),
            is_keyframe=(i % 2 == 0),
            delta_score=float(i),
        )
    db.close()

    # 3 audio segments
    db = TempoGraphDB(db_path)
    db.insert_audio_segment(0, 5000, "someone talks about quokka in the wild")
    db.insert_audio_segment(5000, 10000, "and then a zebrafish appears")
    db.insert_audio_segment(10000, 15000, "the emperor penguin swims away")
    db.close()

    # 4 captions: one with change_line, one with verifier_caption
    db = TempoGraphDB(db_path)
    db.insert_frame_caption(
        0,
        "keys and a phone on a black table",
        change_line="a hand reaches in",
        walker_model="qwen",
    )
    db.insert_frame_caption(
        1, "a zebrafish swims in the tank", change_line=None, walker_model="qwen"
    )
    db.insert_frame_caption(
        2, "the quokka hops along the fence", change_line=None, walker_model="qwen"
    )
    db.insert_frame_caption(
        3,
        "night falls on the savanna",
        change_line=None,
        walker_model="qwen",
    )
    db.save_caption_verdict(3, "night falls over the savanna plain", True, "gemma")
    db.close()

    # 2 detection classes
    db = TempoGraphDB(db_path)
    db.insert_detection(0, 1, "dog", 0.0, 0.0, 0.3, 0.3, 0.95)
    db.insert_detection(0, 2, "person", 0.4, 0.1, 0.9, 0.9, 0.88)
    db.insert_detection(1, 1, "dog", 0.0, 0.0, 0.4, 0.4, 0.92)
    db.close()

    # 1 visual event
    analysis = {
        "entities": [
            {
                "id": "dog_1",
                "type": "dog",
                "first_seen": "00:00",
                "last_seen": "00:01",
                "description": "a brown dog",
            },
        ],
        "visual_events": [
            {
                "type": "approach",
                "start_time": "00:00",
                "end_time": "00:01",
                "description": "dog approaches the feeder",
                "entities": ["dog_1"],
                "confidence": 0.8,
            },
        ],
        "audio_events": [],
        "summary": "A dog approaches the feeder.",
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis))
    return run_dir


# ── Test 1: Index build + counts ────────────────────────────────────────────


class TestIndexBuildCounts:
    def test_build_returns_expected_row_count(self, tmp_path):
        run_dir = _seed_db(tmp_path, "idx")
        n = (
            3 + 6 + 3 + 1
        )  # audio(3) + caption rows(6: 2+1+1+2) + deduped dets(3) + events(1)
        from src.search import build_search_index

        count = build_search_index(run_dir / "tempograph.db")
        assert count == n

    def test_idempotent_no_double_rows(self, tmp_path):
        run_dir = _seed_db(tmp_path, "idx2")
        from src.search import build_search_index

        c1 = build_search_index(run_dir / "tempograph.db")
        c2 = build_search_index(run_dir / "tempograph.db")
        assert c1 == c2
        db = TempoGraphDB(run_dir / "tempograph.db")
        total, _, _ = db.count_frame_captions()
        assert total == 4
        db.close()


# ── Test 2: Every source type findable ─────────────────────────────────────


class TestEverySourceTypeFindable:
    def test_unique_token_per_source(self, tmp_path):
        run_dir = _seed_db(tmp_path, "src")
        from src.search import search

        zebra_hits = search(run_dir / "tempograph.db", "zebrafish")
        assert any(h.source_type == "transcript" for h in zebra_hits)

        quokka_hits = search(run_dir / "tempograph.db", "quokka")
        assert any(h.source_type == "caption" for h in quokka_hits)

        dog_hits = search(run_dir / "tempograph.db", "dog")
        assert any(h.source_type == "detection" for h in dog_hits)

        event_hits = search(run_dir / "tempograph.db", "feeder")
        assert any(h.source_type == "event" for h in event_hits)

    def test_hit_fields(self, tmp_path):
        run_dir = _seed_db(tmp_path, "hit")
        from src.search import search

        hits = search(run_dir / "tempograph.db", "dog")
        dog_hits = [h for h in hits if h.source_type == "detection"]
        for h in dog_hits:
            assert h.source_type == "detection"
            assert h.frame_idx is not None
            assert h.timestamp_ms >= 0

    def test_caption_with_verifier(self, tmp_path):
        run_dir = _seed_db(tmp_path, "verif")
        from src.search import search

        hits = search(run_dir / "tempograph.db", "savanna")
        assert len(hits) >= 1
        v = [h for h in hits if h.source_type == "verifier"][0]
        assert "savanna" in v.snippet.lower()


# ── Test 3: Ranking sanity ──────────────────────────────────────────────────


class TestRankingSanity:
    def test_term_in_three_rows_returns_at_least_three_hits(self, tmp_path):
        run_dir = _seed_db(tmp_path, "rank")
        from src.search import search

        hits = search(run_dir / "tempograph.db", "table")
        assert len(hits) >= 1

    def test_limit_clamps_results(self, tmp_path):
        run_dir = _seed_db(tmp_path, "lim")
        from src.search import search

        hits = search(run_dir / "tempograph.db", "dog", limit=2)
        assert len(hits) <= 2


# ── Test 4: The user's canonical query ─────────────────────────────────────


class TestCanonicalQuery:
    def test_keys_black_table(self, tmp_path):
        run_dir = _seed_db(tmp_path, "canon")
        from src.search import search

        hits = search(run_dir / "tempograph.db", "keys black table")
        caption_hits = [h for h in hits if h.source_type == "caption"]
        assert len(caption_hits) >= 1
        assert any("table" in h.snippet.lower() for h in caption_hits)


# ── Test 5: Robustness ──────────────────────────────────────────────────────


class TestRobustness:
    def test_empty_query_returns_empty(self, tmp_path):
        run_dir = _seed_db(tmp_path, "empty")
        from src.search import search

        assert search(run_dir / "tempograph.db", "") == []

    def test_fts5_operators_no_raise(self, tmp_path):
        run_dir = _seed_db(tmp_path, "ft")
        from src.search import search

        search(run_dir / "tempograph.db", '"dog" AND NOT "cat"')

    def test_missing_frame_captions_table(self, tmp_path):
        run_dir = _seed_db(tmp_path, "legacy")
        db_path = run_dir / "tempograph.db"
        db = TempoGraphDB(db_path)
        db._conn.execute("DROP TABLE IF EXISTS frame_captions")
        db._conn.commit()
        db.close()
        from src.search import build_search_index, search

        count = build_search_index(db_path)
        assert count > 0
        hits = search(db_path, "dog")
        assert isinstance(hits, list)

    def test_no_analysis_json(self, tmp_path):
        run_dir = _seed_db(tmp_path, "noanal")
        db_path = run_dir / "tempograph.db"
        analysis = run_dir / "analysis.json"
        analysis.unlink()
        from src.search import build_search_index, search

        count = build_search_index(db_path)
        assert count > 0


# ── Test 6: AppTest — Search tab ───────────────────────────────────────────


class TestAppTestSearchTab:
    def test_search_tab_with_zebrafish(self, tmp_path):
        run_dir = _seed_db(tmp_path, "app")
        from streamlit.testing.v1 import AppTest

        os.environ["TEMPOGRAPH_RESULTS_DIR"] = str(tmp_path)
        import ui.pages.Results

        at = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py")
        )
        at.run(timeout=60)
        inp = at.text_input(key="search_query")
        assert inp is not None
        inp.set_value("zebrafish")
        at.run(timeout=60)
        buttons = [b for b in at.button if b.key and "find_play" in b.key]
        assert len(buttons) >= 1
        buttons[0].set_value(True)
        at.run(timeout=60)
        assert at.session_state.player_start_s is not None


# ── Test 7: AppTest — no-crash guard ───────────────────────────────────────


class TestAppTestNoCrash:
    def test_empty_run_no_crash(self, tmp_path):
        run_dir = tmp_path / "emptyrun"
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / "tempograph.db"
        db = TempoGraphDB(db_path)
        for i in range(2):
            img = Image.new("RGB", (32, 32), (i, i, i))
            img.save(str(run_dir / f"f{i}.jpg"), format="JPEG")
            db.insert_frame(
                frame_idx=i,
                timestamp_ms=i * 3000,
                image_path=str(run_dir / f"f{i}.jpg"),
                is_keyframe=True,
                delta_score=0.0,
            )
        db.close()
        from streamlit.testing.v1 import AppTest

        import ui.pages.Results

        os.environ["TEMPOGRAPH_RESULTS_DIR"] = str(tmp_path)

        at = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py")
        )
        at.run(timeout=60)
        assert not at.exception
