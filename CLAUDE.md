# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**TempoGraph** — a fully-local multimodal video-analysis pipeline: YOLO26 detection → Depth Anything V2 → Whisper.cpp ASR → chunked Qwen3.5-VL captioning → LLM aggregation. Outputs structured entities, events, and a pyvis graph. Per-run SQLite store + Streamlit UI.

## Quick Commands

```bash
# Install
make install           # pip deps + whisper.cpp build + base.en model
make install-llm       # also: llama.cpp + Qwen3.5-9B (~10 GB)

# Run
make run               # Streamlit UI → http://localhost:8501
make run-cli VIDEO=path/to/clip.mp4   # CLI on one video

# Test
make test              # pytest -v (currently 115 passed)
python3 -m pytest tests/test_foo.py -q   # single file
python3 -m pytest tests/test_foo.py::TestClass::test_method -q   # single test
make smoke             # CPU smoke test (synthetic 5s video, skip-VLM)

# Format / lint
black src/ tests/
flake8 src/ tests/
```

Interpreter: `/home/ashie/anaconda3/bin/python3` (Python 3.12, torch 2.13+cu130, streamlit 1.51).

## Architecture at a Glance

```
source video ──▶ PipelineV2.run() ──▶ results/<name>/
                        │
   Stage 1  Frame selection  (src/modules/frame_selector.py)
   Stage 1.5 Audio transcribe (src/modules/whisper_cpp.py)   [opt-in]
   Stage 2   YOLO detection   (src/modules/detector.py)
   Stage 3   Depth estimation (src/modules/depth.py)          [opt-in]
   Stage 4   Frame scoring    (src/modules/frame_scorer.py)
   Stage 5   VLM captioning   (src/backends/llama_server_backend.py)
   Stage 6   Aggregation      (src/aggregator.py)
```

**Key files:**

| File | Role |
|---|---|
| `src/pipeline_v2.py` | Orchestrator — runs stages sequentially, persists intermediates to SQLite, emits stage events |
| `src/storage.py` | `TempoGraphDB` — SQLite schema + helpers (`run_stages` for resume guards, `run_meta` for key-value) |
| `src/aggregator.py` | `CaptionAggregator` — chunks + transcript → `analysis.json` (entities, events, correlations) |
| `src/models.py` | Pydantic models: `Entity`, `VisualEvent`, `AudioEvent`, `AnalysisResult`, `DetectionBox` |
| `src/summarizer.py` | `generate_summary()` — injectable LLM callable for narrative summaries |
| `src/auto_profile.py` | `probe(path) → VideoFacts` + `derive_plan(facts) → DerivedPlan` — ffprobe-based auto-config |
| `src/rle.py` | COCO-style uncompressed RLE encode/decode (numpy-only, for mask persistence) |
| `src/clip_export.py` | `select_events()` + `export_clips()` — graph-driven event clipping via ffmpeg |
| `src/annotate.py` | `draw_detections()`, `draw_masks()`, `build_annotated_video()` — streamlit-free rendering |
| `src/runtime_estimator.py` | ETA model — per-stage wall-time estimates, calibration from history |
| `ui/app.py` | Streamlit main page — three-screen flow: Landing → Plan → Progress |
| `ui/pages/Results.py` | Streamlit results browser — 8+ tabs (Overview, Frame/Entity Inspector, VLM outputs, Captions, Timeline, Annotated Video, Graph/Clips, Ask, Ethogram, Dataset Export) |

**Storage:** Single SQLite per run at `<run_dir>/tempograph.db`. Tables: `frames`, `detections` (+ `mask_rle`), `depth_frames`, `audio_segments`, `run_stages`, `ethogram_labels`, `run_meta`. Bbox coords are normalised to saved JPEG dimensions (0..1).

**External services:**
- **llama-server** (Qwen3.5-VL): systemd `--user` unit `qwen35-turboquant.service`, port 8082. Pipeline auto-starts/stops via `systemctl`. Every request passes `chat_template_kwargs: {enable_thinking: false}`.
- **whisper.cpp**: built at `~/whisper.cpp`, Vulkan backend, default device 1 (NVIDIA).
- **YOLO26**: `ultralytics` with `yolo26n.pt` weights (auto-download).
- **Depth Anything V2**: `transformers.pipeline("depth-estimation", ...)`.

## Coding Conventions (from AGENTS.md)

- **Python 3.9+**, type hints on all functions, 88-char line width (Black).
- **Imports**: stdlib → third-party → local (`src.`). Alphabetical within groups.
- **Naming**: Classes=PascalCase, functions=snake_case, constants=SCREAMING_SNAKE_CASE, private=underscore prefix.
- **Models**: Pydantic v2, defined in `src/models.py`.
- **Error handling**: explicit exception types, log before raising, handle GPU/CUDA errors gracefully.
- **GPU cleanup**: `gc.collect()` + `torch.cuda.empty_cache()` after model use.
- **Docstrings**: Google format, document exceptions.
- **Tests**: `pytest` in `tests/`, file `test_*.py`, class `Test*`. Run `conftest.py` for fixtures.

## Current State

- **Branch**: `ui-v3-dropflow` (115 tests, zero regressions)
- **Recent work**: Drop→Watch→Explore UI rewrite, resume-from-DB stage guards, mask RLE persistence, graph-driven clip export, summarizer with LLM cache
- **TODO**: See `TODO.md` — items 3+ (cross-run entity registry, Archive-wide Ask, ethogram v2, live mode)
- **Work log**: See `SUMMARY.md` for session-by-session change history

## Known Issues / Footguns

1. **`depth-anything-v2>=1.0.0`** in `requirements.txt` is unsatisfiable on PyPI (only `0.1.0` exists). Pipeline doesn't actually need it.
2. **AMD radv `vk::DeviceLostError`** on Whisper — default is Vulkan device 1 (NVIDIA).
3. **Filename = output dir** — same filename overwrites previous run results.
4. **4 GB Streamlit upload** — Python holds entire upload in RAM during `uploaded.read()`.
5. **Mask persistence** — `mask_rle` column added but seg variant masks not yet fully wired into all UI paths.
6. **Pre-2026-04-27 bboxes** — old runs have bboxes normalised against source video dims (shrunk by resize scale). Re-run to fix.
7. **LLM backend URL** — hardcoded `http://127.0.0.1:8082` in summarizer/Results. Should read from config.
8. **ETA calibration** — `record_stage_timing()` exists but isn't wired into `pipeline_v2.run()`.

## GPU Work Policy

- GPU work only on CUDA device 0 (RTX 3060) unless explicitly instructed otherwise.
- The AMD 9070 XT serves the Qwen3.5-VL model — **do not allocate memory on it** (it's the model running you).

## Important Files to Check Before Editing

- `TODO.md` — next tasks, constraints, environment facts
- `SUMMARY.md` — recent changes, acceptance test outputs, known gaps
- `AGENTS.md` — full coding standards and conventions
- `docs/PIPELINE.md` — deep stage-by-stage internals (last verified 2026-04-27)
- `src/pipeline_v2.py` — the orchestrator; understand stage flow before touching it
