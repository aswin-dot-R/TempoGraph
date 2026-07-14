# PS6b — Natural-language search: TEST SUITE (9B lane)

Tests for `src/search.py` + the Search tab, written in parallel with the
implementation (`docs/ps/ps6a.md`). Code against the frozen contracts in
that file's "FROZEN CONTRACTS" section — do not read or wait for the
implementation. Failing until it lands is expected; your gate is clean
collection.

## Scope fence

- **Files you may create:** `tests/test_search.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.
Top of file: `pytest.importorskip("src.search")`.

## Fixture recipe

Build run DBs with `TempoGraphDB` (see `tests/test_dense_walker.py` for
the pattern): insert frames (with timestamps), audio segments,
frame_captions rows (`insert_frame_caption` + `save_caption_verdict`),
detections (`insert_detection`). For event hits, write an
`analysis.json` beside the DB with one visual_event
(`{"visual_events": [{"type": "approach", "description": "bird lands on
the feeder", "start_time": "00:12", ...}]}`). AppTest fixtures: copy the
`TEMPOGRAPH_RESULTS_DIR` pattern from `tests/test_results_apptest.py`.

## Tests

1. **Index build + counts** — fixture with 3 transcript segments, 4
   captions (one with change_line, one with verifier_caption), 2
   detection classes, 1 event → `build_search_index` returns the
   expected row count; calling it twice doesn't double rows.
2. **Every source type findable** — plant a unique token per source
   ("zebrafish" only in a caption, "quokka" only in transcript, etc.);
   `search` returns exactly the right hit with correct `source_type`,
   `timestamp_ms`, and frame_idx (None where contract says so).
3. **Ranking sanity** — a term appearing in 3 rows returns ≥3 hits;
   `limit=2` returns 2.
4. **The user's canonical query** — captions include "keys and a phone
   on a black table"; `search(db, "keys black table")` (terms OR/AND per
   contract) returns that caption as a hit.
5. **Robustness** — empty query → []; query with FTS5 operators
   (`"dog" AND NOT`) does not raise; missing frame_captions table (a
   legacy-style DB with only frames+audio) still indexes what exists.
6. **AppTest: Search tab** — fixture run; type "zebrafish" into
   `search_query` via `at.text_input(key="search_query").set_value(...).run()`;
   at least one `find_play_` button exists; clicking it sets
   `at.session_state["player_start_s"]` to the hit's seconds.
7. **AppTest: no-crash guard** — run DB with zero indexable rows →
   Search tab renders without exception.

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_search.py --collect-only -q
/home/ashie/anaconda3/bin/python3 -m black tests/test_search.py
git status --porcelain   # only tests/test_search.py
```

(do not commit — gate reviewer merges both lanes)
