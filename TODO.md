# TempoGraph TODO

Tracking forward work for the v2-only branch. Triaged by effort × value.
File:line references point at where each change would land.

Last updated: 2026-04-28.

---

## Tier 1 — small effort, real value (do first)

### Wire `dataset_exporter` into the UI
- **Status**: module shipped on `52b9217`, never wired to UI
- **Code**: `src/dataset_exporter.py` already has `export_coco_annotations`,
  `export_captions_jsonl`, `export_frame_captions_jsonl`, `export_all`
- **Add**: new "Dataset" tab in `ui/pages/Results.py` with:
  - "Build COCO + JSONL" button → calls `export_all(run_dir)`
  - class-distribution histogram
  - sample image per class grid
  - download-zip button for `<run_dir>/exports/`
- **Effort**: ~1 hour, ~80 LOC

### Per-frame transcript join in Frame inspector
- **Status**: not started
- **Code**: `ui/pages/Results.py:_render_frame_inspector()` (around line 333)
- **Add**: SQL join of `audio_segments` against the current frame's
  `timestamp_ms` to show what was being said at that moment, in a
  "Transcript at this timestamp" box next to the detections table
- **Query already shown in `docs/PIPELINE.md:Storage schema`**:
  ```sql
  SELECT a.text FROM audio_segments a JOIN frames f
    ON a.start_ms <= f.timestamp_ms AND a.end_ms > f.timestamp_ms
    WHERE f.frame_idx = ?;
  ```
- **Effort**: ~20 LOC

### Stop button mid-run
- **Status**: not started; today you can only kill the streamlit process
- **Approach**: `threading.Event` passed into `PipelineV2.run()`, checked
  between stages and inside the chunk loop
- **Code**: `src/pipeline_v2.py:PipelineV2.__init__` add `cancel_event`
  param; `_stage()` calls check it; `caption_chunks` callback can also
  check it. `ui/app.py` wires a "Cancel" button that sets the event.
- **Effort**: ~30 LOC, requires care that partial DB writes don't leave
  the run in an unrecoverable state (it doesn't — DB rows are
  per-frame-idempotent, but `chunks.json` would be partial; that's
  acceptable since the user explicitly asked for stop)

### Calibrate ETA from past runs
- **Status**: not started; ETA constants in `src/runtime_estimator.py`
  are hand-tuned and the Jurassic Park run blew past by 17 min
- **Approach**: append `(stage_name, elapsed_s, n_input_units)` to
  `results/.calibration.jsonl` after each run. Estimator reads the
  most recent N entries per stage and fits a simple linear model.
- **Effort**: ~50 LOC. Self-improving every time you run.

---

## Tier 2 — medium effort, real value

### Mask persistence (so the seg variant pays its way)
- **Status**: seg variant computes masks every frame and discards them
- **Schema change** (`src/storage.py`):
  ```sql
  ALTER TABLE detections ADD COLUMN mask_rle TEXT;
  -- OR: mask_npy_path TEXT (lighter on the DB, heavier on disk)
  ```
- **Code**: `src/modules/detector.py:detect_to_db` — when seg model is
  loaded, encode each result's mask polygon to COCO RLE (use
  `pycocotools` if available, else simple run-length) and store
- **Bonus**: render masks in the Results page Annotated video tab and
  Frame inspector overlay
- **Effort**: ~80 LOC + new dep (`pycocotools` for RLE)

### Resume-from-DB on crash
- **Status**: long runs fail and start from scratch
- **Approach**: at run start, check `db.count_frames()`,
  `db.count_detections()`, `db.count_audio_segments()`. If a stage's
  output already exists for the input set, skip that stage.
- **Code**: `src/pipeline_v2.py:run()` — guard each stage's
  `_stage("X", "start")` block on a presence check
- **Risk**: distinguishing "already done" from "partial run that
  crashed mid-stage". Solution: write a `runs(run_id, stage,
  finished_at)` table that explicitly records stage completion.
- **Effort**: ~60 LOC + 1 new table

### GitHub Actions CI
- **Status**: nothing on push/PR currently
- **Add**: `.github/workflows/ci.yml`:
  - matrix Python 3.10/3.11/3.12
  - install (no whisper.cpp build — too slow for CI; mock it via
    pytest fixtures)
  - run `make test`
- **Effort**: ~30 LOC YAML

### Multimodal correlations viz
- **Status**: aggregator produces `multimodal_correlations` array but
  Results page never displays it
- **Add**: a sub-section in the Overview tab that renders the audio↔visual
  links as connector lines on the existing Plotly Gantt timeline
- **Code**: `ui/pages/Results.py:_render_events_timeline` add
  `add_shape(...)` for each correlation
- **Effort**: ~80 LOC

### YOLO bbox bug back-fill
- **Status**: runs from before 2026-04-27 have bboxes normalised against
  source video dims instead of saved JPEG dims
- **Fix**: write a one-shot script `scripts/fix_legacy_bboxes.py` that
  reads each old run's `tempograph.db`, looks up the saved JPEG width,
  and rescales the rows
- **Effort**: ~40 LOC

---

## Tier 3 — bigger projects

### Dataset generation mode (UI + class filter + YOLO-World)
- **Status**: discussed, partial backend (`dataset_exporter.py` exists)
- **Scope**: sidebar mode toggle (Analysis / Dataset generation), class
  selector (closed = COCO-80, open = YOLO-World text classes), filter
  flags (min confidence, min frames per class, include negatives), train/
  val/test split, dedicated Results-page "Dataset" tab with class
  distribution histogram + sample-per-class grid + zip download
- **Effort**: half-day, ~250 LOC across `ui/app.py`,
  `ui/pages/Results.py`, `src/modules/detector.py` (class filter via
  `model.predict(classes=[...])` for closed-vocab,
  `model.set_classes([...])` for YOLO-World), `src/dataset_exporter.py`

### VLM-as-classifier mode for novel classes
- **Status**: discussed
- **Scope**: per-frame call to qwen with prompt "From this list [X, Y, Z],
  which are present? Output JSON." Slow (~5 s/frame) but most accurate
  open-vocab labels with no bbox.
- **Effort**: ~80 LOC, ~50× slower than YOLO so should be a separate
  detector mode (not the default)

### SAM masks for novel classes
- **Pattern**: YOLO-World detects bboxes → feed each into Segment
  Anything for the mask. Two-pass, slower, but works for any vocabulary.
- **Effort**: ~120 LOC + SAM model dep (~600 MB) + extra inference cost

### Docker image for v2-only
- **Status**: v1 Dockerfiles deleted on this branch
- **Scope**: new `Dockerfile` that bakes in conda env + whisper.cpp
  build + Streamlit. Optional second image with llama.cpp + Qwen
  weights baked in (~13 GB image — only worth it if shipping to
  non-experts).
- **Effort**: ~150 LOC + a few hours of build-and-test cycles

### PyPI package
- **Scope**: `pyproject.toml` for `pip install tempograph`. Doesn't
  cover whisper.cpp or llama-server (still need bootstrap), but
  simplifies the Python-side install.
- **Effort**: ~50 LOC + first-time package publishing setup

---

## Known issues / footguns

These don't block anything but should be on the radar:

1. **`requirements.txt` openai-whisper / depth-anything-v2 lines are
   commented out** — neither is needed for v2 (whisper.cpp + transformers
   replace them). If anyone uncomments them, install will fail on
   `depth-anything-v2>=1.0.0` (PyPI only has 0.1.0).
2. **AMD radv `vk::DeviceLostError`** intermittently when whisper-cli
   targets Vulkan device 0. Pipeline auto-falls-back to NVIDIA → CPU now,
   but it's still a flake. Worth filing upstream against radv if it keeps
   happening.
3. **Filename-keyed output dirs** — uploading two videos with the same
   filename overwrites the first run. Hash-suffix the output dir if it
   exists to avoid silent data loss.
4. **Streamlit upload limit raised to 4 GB** via `.streamlit/config.toml`.
   The whole upload sits in Python RAM during `uploaded.read()`. Switch
   to streaming `uploaded.read(chunk_size)` if memory ever becomes an
   issue.
5. **Mask discard** — the seg variant of YOLO computes instance masks
   but they aren't persisted (no `mask_*` columns). See Tier 2 above.
6. **Pre-2026-04-27 bbox bug** — runs from before that date have bboxes
   normalised against source video dims instead of saved JPEG dims. See
   Tier 2 back-fill script.
7. **Aggregator + 100k context** — single-pass aggregator works up to
   30 chunks (`single_pass_max_chunks=30`); beyond that switches to
   hierarchical compression in `_compress_hierarchical()`. Watch the
   aggregator row in the live context-window panel; if it goes
   orange/red, hierarchical is already saving you.
8. **Service unit hard-codes LM Studio paths** — `qwen35-turboquant.service`
   ExecStart points at `/home/ashie/.lmstudio/models/...`. Brittle if
   LM Studio reorganises. `bootstrap.sh --with-llm` writes a separate,
   independent unit (`qwen-tempograph.service`) — recommend migrating.

---

## Recently shipped (last week)

- v2-only branch: legacy pipeline removed, one-line installer
  (`Makefile` + `bootstrap.sh` + `QUICKSTART.md`)
- VLM frame dedup with keyframe-protection (Stage 4.5)
- Entity-ID persistence across chunks (registry + `NEW_ENTITIES:` line)
- Whisper GPU fallback chain (NVIDIA → AMD → CPU)
- Dataset exporter module (COCO + JSONL) — *not yet wired to UI*
- Batch runner module — *CLI only, no UI yet*
- Live elapsed timer + ETA + per-stage cost breakdown in UI
- Live VLM context-window panel (per-chunk token usage with peak indicator)
- Interactive timeline with click-to-seek on the annotated video
- `docs/PIPELINE.md` (1011 lines) — deep technical doc
- Bbox normalisation bug fix (saved JPEG dims, not source video dims)
- Depth backend swap from `depth-anything-v2` PyPI pkg → `transformers`
  pipeline (auto-downloads weights from HF)
