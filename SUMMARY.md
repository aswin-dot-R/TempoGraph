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

---

## 2026-07-11 — TODO items 1 & 2: mask persistence + graph-driven clip export

Branch `ui-v3-dropflow`. Suite went 68 → 115 passed (47 new tests), zero
regressions. Interpreter `/home/ashie/anaconda3/bin/python3` throughout.

### Item 1 — Mask persistence (seg models pay their way)

- **Schema**: `detections.mask_rle TEXT` added to `src/storage.py` SCHEMA plus
  a guarded `PRAGMA table_info` → `ALTER TABLE` migration in
  `TempoGraphDB.__init__` (legacy DBs migrate on open; NULL = no mask;
  idempotent across reopens).
- **Encoding**: new pure `src/rle.py` (numpy only, no new deps) — COCO-style
  uncompressed RLE: column-major flatten, counts start with the zero-run,
  `size=[H,W]`; JSON-string form persisted in the column. In
  `ObjectDetector.detect_to_db`, seg-variant results rasterise
  `result.masks.xyn` polygons (normalised → alignment-safe under letterbox)
  at the saved-frame resolution and store one RLE per detection.
- **Rendering**: new streamlit-free `src/annotate.py`
  (`draw_detections`/`draw_masks`/`apply_depth_overlay`/
  `build_annotated_video`) shared by `ui/pages/Results.py` and the new
  `scripts/annotate_video.py` CLI. Frame inspector gained a **Masks**
  checkbox (semi-transparent fill, per-entity colour keyed on track id, on
  by default when the run has masks); the annotated-video tab gained a Masks
  toggle and the CLI carries the `--masks` flag.
- Also fixed a latent AppTest-visible crash: the overview and entity-inspector
  event timelines produced duplicate auto-generated plotly IDs (now keyed),
  and `RESULTS_DIR` is overridable via `TEMPOGRAPH_RESULTS_DIR` for fixtures.

**Acceptance (pasted):**

```
$ python3 -m pytest tests/test_rle.py -q                 # RLE round-trip property tests
11 passed in 1.02s
$ python3 -m pytest tests/test_mask_persistence.py -q    # migration + CPU seg smoke (5 s testsrc)
7 passed in 3.34s
$ python3 -m pytest tests/test_results_apptest.py -q     # AppTest: inspector on masked fixture DB
6 passed in 4.24s

# CPU seg smoke, 5 s ffmpeg testsrc clip, yolo26n-seg.pt (repo root):
seg smoke run: 21 detections, 21 with non-NULL mask_rle
$ python3 scripts/annotate_video.py <segrun> --masks --min-conf 0.01
wrote .../segrun/annotated_strip.mp4 (0.10 MB)
```

Note: `testsrc` is an abstract pattern, so the smoke run uses conf 0.01 —
the (spurious but deterministic) instances are exactly what is needed to
prove masks flow detector → RLE → DB → overlay. `yolo26n-seg.pt` was copied
from `~/TempoGraph` into the worktree root (gitignored via `*.pt`).

### Item 2 — Graph-driven clip export

- **`src/clip_export.py`**:
  `select_events(db, entity=None, behavior=None, time_range=None)` accepts a
  `tempograph.db` path or run dir; events come from `analysis.json`
  `visual_events` (the graph's edges) plus contiguous same-behavior
  `ethogram_labels` runs in the DB (ethogram spans are skipped when an entity
  filter is set, as they are not entity-scoped). Each event is padded ±1.5 s
  (start clamped at 0) and overlapping/touching spans merge with
  " + "-joined labels — the pure `pad_and_merge` helper is unit-tested
  directly. `export_clips(video_path, spans, out_dir, montage=False)` cuts
  one mp4 per span: stream-copy first, ffprobe-verified, accurate re-encode
  fallback when keyframes misalign; `montage=True` concatenates with `xfade`
  crossfades (0.25 s) and burned-in `drawtext` label lower-thirds (label-less
  fallback if drawtext is unavailable).
- **UI**: new **Graph** tab in `ui/pages/Results.py` hosting the pyvis entity
  graph (previously only under Files) and the **Clips** section:
  entity/behavior pickers built from existing graph data, live span-preview
  table, source-video path input (defaults from the new `run_meta.video_path`
  that `pipeline_v2` now persists), Export button, montage toggle, and
  per-clip download buttons.

**Acceptance (pasted):**

```
$ python3 -m pytest tests/test_clip_export.py -q   # span math units + ffmpeg integration
23 passed in 7.06s

# End-to-end on a fixture run + 14 s ffmpeg testsrc source:
spans: [(500, 4500, 'approach: dog_1'), (6500, 13500, 'interact: dog_1, person_1 + grooming')]
clip_000_approach_dog_1.mp4: want 4.00s got 4.00s (delta 0.000s)
clip_001_interact_dog_1_person_1_grooming.mp4: want 7.00s got 7.00s (delta 0.000s)
montage: montage.mp4 duration 10.77s size 116 kB   # 4 + 7 − 0.25 s crossfade, ffprobe-valid h264
```

### Full suite

```
$ /home/ashie/anaconda3/bin/python3 -m pytest -q
115 passed, 21 warnings in 16.99s
```

Commits: `b6b0353` (item 1), `a1c92cc` (item 2).

---

## 2026-07-12 — PS1: frame_captions schema + WAL + DB helpers

Executed by Ornith 35B architect (opencode); gate-reviewed by Fable.
Suite 115 → 121 passed. `frame_captions` table (walker INSERTs, verifier
UPDATEs), `idx_fc_escalated`, WAL + busy_timeout for the two concurrent
writers, five helpers (`insert_frame_caption`, `fetch_unverified_escalations`,
`save_caption_verdict`, `count_frame_captions`, `get_frame_caption`).
Review fixes by Fable: hoisted datetime import to module level, corrected
`get_frame_caption` docstring (returns dict), black-formatted.

```
$ pytest tests/test_dense_schema.py -q  → 6 passed
$ pytest -q                             → 121 passed
```

---

## 2026-07-12 — PS2: DenseCaptionWalker (9B per-frame captions + escalation)

Executed by Ornith 35B architect (opencode); gate-reviewed by Fable.
Suite 121 → 142 passed (21 new tests). `src/modules/dense_captioner.py`:
`parse_two_lines`, `jaccard`, `should_escalate`, pure-python `_percentile`,
and `DenseCaptionWalker.walk()` — resumable, cancellable, per-frame error
tolerance, prev-caption threading through skips, retrieval-grade prompt.
Review fix by Fable: unreadable frame JPEG now counts as an error and the
walk continues (was: crash before the HTTP try block); black-formatted.

```
$ pytest tests/test_dense_walker.py -q  → 21 passed
$ pytest -q                             → 142 passed
```

---

## 2026-07-12 — PS3: EscalationVerifier + parallel orchestrator

Executed by Ornith 35B architect (opencode); gate-reviewed by Fable.
Suite 142 → 153 passed (11 new tests). `EscalationVerifier` (35B second
opinions with AGREE/DISAGREE verdicts, 3-retry cap, poll loop) and
`run_dense_captioning()` — verifier daemon thread starts BEFORE the walk,
overlap proven by test. Gate-review fix by Fable: capped rows stayed
unverified in the DB and shadowed later rows behind `LIMIT batch_size`,
and the all-capped exit could kill the verifier while the walker was
still producing escalations (e.g. brief 35B outage → 3 fast retries per
early frame → premature exit). Now over-fetches by the capped count,
filters in Python, and only exits via the walker-done drain path. Noisy
per-poll info logs downgraded to debug; black-formatted.

```
$ pytest tests/test_dense_verifier.py -q  → 11 passed
$ pytest -q                               → 153 passed
```

---

## 2026-07-12 — PS4: dense captioning wired into pipeline, UI, aggregation

Executed by Ornith 35B architect (opencode); gate-reviewed by Fable.
Suite 153 → 160 passed (7 new). Pipeline gains the opt-in "Dense captions"
stage (after YOLO, resume-guarded, cancel-aware, progress-forwarded);
`CaptionAggregator.load_dense_timeline()` (escalated rows kept, verifier
caption preferred on disagreement, even subsampling to ≤120 lines) feeds
a `[dense captions]` block into the aggregation prompt and a
`dense_timeline` key into analysis.json; frame inspector shows walker
caption + change line + 35B second opinion (✅/⚠️); Timeline gains a dense
caption section. Gate-review fix by Fable: the dense block was prepended
to the LLM prompt TWICE in the `_single_pass` path (once into log_lines,
once by `_single_pass_from_text`) — now prepended exactly once.
black-formatted all four files.

```
$ pytest tests/test_dense_pipeline.py -q  → 7 passed
$ pytest -q                               → 160 passed
$ streamlit headless boot                 → UI_BOOTS
```

Dense Temporal Captioning (PS1–PS4) is now complete end-to-end.

---

## 2026-07-12 — HOTFIX 1: real-video drops no longer crash (parallel lanes)

First two-lane parallel execution: Ornith 35B implemented (hotfix1a),
Ornith 9B wrote the resilience tests (hotfix1b) simultaneously against
frozen contracts; Fable gate-reviewed and merged. Suite 160 → 166.

Root cause (reproduced): real 1080p video → derive_plan enables depth →
`from transformers import pipeline` → ModuleNotFoundError kills the run.
Fix: derive_plan only offers depth when transformers is importable
(plan gains `notes`, e.g. "depth off: transformers not installed");
Depth / Dense captions / Audio stage bodies armored — failure emits a
stage "error" event and the run continues.

Gate-review fix by Fable: 9B's test helper accepted `dense_captions` but
never forwarded it to PipelineV2, so the dense resilience test asserted
against a skipped stage (1 failed → fixed → 6/6). 35B legitimately
stepped outside its fence to update `tests/test_auto_profile.py` (the
depth assertions are now environment-aware) — fence was wrong, change
accepted.

```
$ pytest tests/test_optional_stage_resilience.py -q → 6 passed
$ pytest -q                                         → 166 passed
$ python -m src.pipeline_v2 --video file_example... → Done in 4.0s (was: crash)
$ derive_plan(probe(...)) → depth_enabled: False, notes: ['depth off: transformers not installed']
```

---

## 2026-07-13 — PS5: click-to-play (parallel lanes, Fable-driven)

First fully Fable-orchestrated parallel item: 35B implemented (ps5a), 9B
wrote 16 tests (ps5b) simultaneously; 35B then ran the gate seam-fix from
Fable's failure diagnosis. All 6 combined-run failures were TEST-lane
bugs (1-based segment_id assumption, nonexistent frame index, unbuilt
250-segment fixture, unguarded key=None); the implementation was
contract-correct throughout. Suite 166 → 182. ui/video_player.py
(resolve_video / make_strip_mapper / render_player) + every timestamp
surface in Results.py wired to one sticky player with pagination.

```
$ pytest -q → 182 passed
$ streamlit headless boot → UI_BOOTS
```

---

## 2026-07-13 — PS6: natural-language search (parallel lanes, Fable-driven)

35B implemented src/search.py (FTS5 index over transcript + dense
captions + change lines + verifier opinions + detection classes +
analysis.json events; bm25; query sanitising; E2B :8093 query rewrite
with graceful fallback) + Search tab with show-frame / play-span actions.
9B wrote 14 tests. Gate seam-fix by 35B from Fable's diagnosis: 13
failures were one invented kwarg in the test fixture
(insert_frame_caption(verifier_caption=...) → real two-call API), plus
str/str path bugs and key=None guards. Suite 182 → 196.

```
$ pytest -q --ignore=tests/test_highlight_reel.py → 196 passed
```
(highlight_reel tests excluded: 9B mid-write on PS7b at gate time)

---

## 2026-07-13 — PS7: highlight reel (parallel lanes, Fable-driven)

35B: src/highlight_reel.py (greedy pick_highlight_spans by delta_score
with min-gap + padding + merge; build_highlight_reel via ffmpeg with
xfade, re-encode accuracy, single-span no-fade path, ValueError on empty
spans) + Overview "Highlight Reel" card with cached-reel reuse. 9B: 10
tests incl. hand-checked greedy math and ffprobe-verified integration.
Gate seam-fix (35B): empty-spans ValueError ordering, tmp_path/dir
fixture bugs, ms-vs-seconds assertion. Suite 196 → 206.

```
$ pytest -q → 206 passed
```
