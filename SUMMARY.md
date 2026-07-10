# SUMMARY — Drop → Watch → Explore UI Implementation

## Follow-up (resume + summarizer)

### What Was Done

**Task 1 — Resume-from-DB stage guards**
- `src/pipeline_v2.py:run()` now checks `is_stage_complete()` before every stage
- Completed stages are skipped on resume with a log message
- After each stage completes, `mark_stage_complete()` records the result
- `vlm_frames` are persisted in `run_meta` for resume across Frame scoring → VLM captioning
- Helper methods: `_load_frame_paths_from_db()`, `_load_selection_from_db()`, `_save_vlm_frames()`, `_load_vlm_frames()`
- `src/storage.py`: new `run_meta` table + `get_meta()` / `set_meta()` / `delete_meta()`

**Task 2 — Wire summarizer into Results Overview tab**
- `ui/pages/Results.py:_render_summary()` now calls `_generate_run_summary()`
- LLM callable resolution via 2-second HTTP health probe to `/v1/models`
- If reachable, calls `llama-server /v1/chat/completions`; otherwise uses heuristic
- Generated summary cached in `run_meta` table — second render returns cached value
- Heuristic path: 5-line narrative from entities, events, audio, summary text

### Acceptance Test Output

```
$ /home/ashie/anaconda3/bin/python3 -m pytest -q
68 passed, 21 warnings

$ /home/ashie/anaconda3/bin/python3 -m pytest tests/test_resume.py -q
3 passed

$ /home/ashie/anaconda3/bin/python3 -m pytest tests/test_summarizer.py -q
8 passed

$ grep -q "is_stage_complete" src/pipeline_v2.py
→ GUARDS_WIRED ✓

$ grep -qE "summarizer|generate_summary" ui/pages/Results.py
→ SUMMARIZER WIRED ✓

$ timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
    --server.headless true --server.port 8599
→ UI_BOOTS ✓
```

### Known Gaps

1. **Deep resume (partial pipeline)** — When resuming from mid-pipeline, some downstream variables (e.g., `selection` object) need to be reconstructed from DB. The current implementation handles Frame selection, YOLO detection, and VLM captioning resume. Edge cases around Depth estimation resume with non-contiguous frame indices are not exhaustively tested.

2. **LLM backend URL is hardcoded** — `_llm_health_probe()` defaults to `http://127.0.0.1:8082`. In production, this should read from the config or session state.

3. **ETA calibration not wired** — `record_stage_timing()` exists in `runtime_estimator.py` but isn't called from `pipeline_v2.run()`. This remains a TODO.

## What Changed

### New Files

| File | Description |
|---|---|
| `src/auto_profile.py` | `probe(path) → VideoFacts` + `derive_plan(facts) → DerivedPlan` — pure functions that probe video metadata via ffprobe and derive pipeline knobs per the design doc rules |
| `src/summarizer.py` | `generate_summary()` — injectable LLM callable for narrative summaries; defaults to heuristic-based if no LLM provided |
| `scripts/smoke_dropflow.py` | CPU end-to-end smoke test: 5s synthetic video, skip-vlm, asserts DB exists with frames and detections table |
| `tests/test_resume.py` | Integration tests for resume-from-DB stage guards (3 tests) |
| `tests/test_summarizer.py` | Tests for summarizer generation, heuristic fallback, and caching (8 tests) |

### Modified Files

| File | Changes |
|---|---|
| `ui/app.py` | Complete rewrite: three-screen flow (Landing → Plan → Progress). Landing has drop zone + path input + recent runs gallery with **zero sidebar control widgets**. Plan screen shows derived plan sentence + ETA + **Analyze** button + collapsed **Adjust plan** expander with legacy knobs pre-filled. Progress screen shows stage checklist + cancel + live counters. |
| `ui/pages/Results.py` | Added "Ask" tab with SQL-grounded Q&A (`_render_ask_tab`, `_answer_question`). Per-frame transcript join already existed (was in `_render_frame_inspector`). Dataset export tab already existed. **Follow-up:** Added narrative summary section to Overview tab with LLM health-probe resolution and DB-backed caching (`_generate_run_summary`, `_llm_health_probe`, `_llm_call`). |
| `tests/test_pipeline_v2.py` | Fixed: test was checking `caption_chunks` but `dynamic_chunking=True` by default, so `caption_frames_dynamic` is actually called. Added mock for `caption_frames_dynamic`. |
| `src/pipeline_v2.py` | **Follow-up:** Added resume-from-DB stage guards before every pipeline stage. `is_stage_complete()` checks skip completed stages; `mark_stage_complete()` records completion. Added `run_meta` table helper methods (`get_meta`, `set_meta`, `delete_meta`). Persisted `vlm_frames` via `run_meta` for resume across Frame scoring / VLM dedup stages. |
| `src/storage.py` | **Follow-up:** Added `run_meta` table schema + `get_meta()`, `set_meta()`, `delete_meta()` methods to `TempoGraphDB`. |

### Existing Code (unchanged but relied upon)

- `src/pipeline_v2.py` — already has `cancel_event` param, `_stage` checks, `run_stages` table
- `src/runtime_estimator.py` — already has `record_stage_timing` and `load_calibration` for ETA calibration
- `src/storage.py` — already has `run_stages` table with `mark_stage_complete`, `is_stage_complete` for resume-from-DB
- `ui/pages/Results.py:_render_frame_inspector` — already has per-frame transcript join via audio_segments timestamp overlap

## What Was Verified

### Acceptance Tests — All Pass

```
$PY -m pytest -x -q                          → 57 passed
$PY -m pytest tests/test_ui_dropflow.py -q   → 15 passed
$PY -m pytest tests/test_auto_profile.py -q  → 13 passed
$PY scripts/smoke_dropflow.py                → PASS: 5 frames, detections table exists
timeout 12 $PY -m streamlit run ui/app.py ... → UI_BOOTS
```

### Part B Items

1. **Cancel event** — Already implemented in `pipeline_v2.py:72-73, 112-115`. The UI's progress screen creates a `threading.Event` and passes it to `PipelineV2`. The `_stage` method checks `is_set()` before each stage.

2. **ETA calibration** — Already implemented in `runtime_estimator.py:166-231`. `record_stage_timing()` appends to `results/.calibration.jsonl`; `load_calibration()` fits a linear model from history.

3. **Per-frame transcript join** — Already implemented in `Results.py:468-478`. Shows audio segments whose `[start_ms, end_ms]` overlaps the frame's `timestamp_ms`.

4. **Resume-from-DB** — Schema supports it (`run_stages` table exists in `storage.py:49-54`). Guards would need to be added to `pipeline_v2.run()` before each stage's `_stage("X", "start")` call. Not yet wired into the pipeline code — the infrastructure is there.

5. **Mask RLE persistence** — Skipped (stretch goal). Would require ALTER TABLE on detections.

## Known Gaps

1. **Resume-from-DB guards** — The `run_stages` table exists but `pipeline_v2.run()` doesn't check it before running stages. This is a TODO in TODO.md (Tier 2, ~60 LOC).

2. **ETA calibration not wired into pipeline** — `record_stage_timing()` exists but isn't called from `pipeline_v2.run()`. Would need to add calls after each stage completes.

3. **Summarizer LLM integration** — `src/summarizer.py` has an injectable `llm_callable` parameter but the UI/Results page doesn't yet wire it to the llama-server backend. In production, this would call the same llama-server URL.

4. **Ask tab uses rule-based answers** — Currently `_answer_question()` uses pattern matching on the SQLite data. It doesn't call an LLM. This is sufficient for the acceptance criteria but could be enhanced with LLM answers.

5. **Graceful stage failure** — The PS asks for "banner + continue, never a traceback in the UI". The current progress screen wraps the pipeline in try/except and shows an error, but doesn't persist partial results in a recoverable way. The `run_stages` table infrastructure supports this but isn't wired yet.

## Screenshots Not Required (per PS)

## Commands That Were Run

```bash
# 1. Full test suite
/home/ashie/anaconda3/bin/python3 -m pytest -x -q           # 57 passed

# 2. UI contract tests
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_ui_dropflow.py -q  # 15 passed

# 3. Auto-profile tests
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_auto_profile.py -q  # 13 passed

# 4. CPU smoke test (exact PS spec)
ffmpeg -y -f lavfi -i testsrc=duration=5:size=640x360:rate=10 /tmp/tg_smoke.mp4
/home/ashie/anaconda3/bin/python3 scripts/smoke_dropflow.py /tmp/tg_smoke.mp4
# Output: PASS: 5 frames, detections table exists

# 5. Streamlit headless boot
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599
curl -sf localhost:8599 && echo UI_BOOTS   # UI_BOOTS
```
