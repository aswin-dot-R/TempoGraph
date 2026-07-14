# TempoGraph — Agent Coding Guidelines

## Quick Commands

```bash
# Install
make install           # pip deps + whisper.cpp + base.en model
make install-llm       # also: llama.cpp + Qwen3.5-9B (~10 GB)

# Run
make run               # Streamlit UI → http://localhost:8501
make run-cli VIDEO=clip.mp4   # CLI on one video

# Test / lint
make test              # pytest -v (currently 265 passed)
black src/ tests/
flake8 src/ tests/

# Smoke (no GPU/LVM required)
make smoke             # CPU smoke test on synthetic 10s video, skip-VLM
```

Interpreter: `python3` (Python 3.12).

## Architecture at a Glance

```
source video ──▶ PipelineV2.run() ──▶ results/<name>/
                        │
    Stage 1   Frame selection        (src/modules/frame_selector.py)
    Stage 1.5 Audio transcript       (src/modules/whisper_cpp.py)   [opt-in]
    Stage 2   YOLO26 detection       (src/modules/detector.py)
    Stage 3   Depth estimation       (src/modules/depth.py)         [opt-in]
    Stage 4   Frame scoring          (src/modules/frame_scorer.py)
    Stage 5   Dense captions (9B)    (src/modules/dense_captioner.py) + EscalationVerifier
    Stage 6   VLM captioning (chunked)  (src/backends/llama_server_backend.py)
    Stage 7   Aggregation            (src/aggregator.py)
```

**Key files:**

| File | Role |
|---|---|
| `src/pipeline_v2.py` | Orchestrator — runs stages sequentially, persists to SQLite, resume-guarded |
| `src/storage.py` | `TempoGraphDB` — SQLite schema + helpers (WAL + busy_timeout) |
| `src/aggregator.py` | `CaptionAggregator` — chunks + transcript → `analysis.json` |
| `src/models.py` | Pydantic models: `Entity`, `VisualEvent`, `AudioEvent`, `AnalysisResult`, `DetectionBox` |
| `src/settings.py` | `TEMPOGRAPH_*` env vars (zero-dep, re-reads each call) |
| `src/auto_profile.py` | `probe(path) → VideoFacts` + `derive_plan(facts) → DerivedPlan` |
| `src/summarizer.py` | `generate_summary()` — injectable LLM callable for narratives |
| `src/search.py` | FTS5 search over transcript + captions + events (BM25) |
| `src/highlight_reel.py` | `pick_highlight_spans()` + `build_highlight_reel()` via ffmpeg |
| `src/clip_export.py` | Graph-driven clip export |
| `src/annotate.py` | Streamlit-free detection/mask rendering |
| `src/rle.py` | COCO-style RLE encode/decode (numpy-only) |
| `ui/app.py` | Main page — three-screen flow: Landing → Plan → Progress |
| `ui/pages/Results.py` | Results browser — 8+ tabs (Overview, Inspector, VLM, Captions, Timeline, Graph, Clips, Search, etc.) |
| `ui/live_view.py` | Real-time dense-captioning view |
| `ui/theme.py` | Dark-first CSS theme |
| `ui/video_player.py` | Click-to-play video player |

**Storage:** Per-run SQLite at `<run_dir>/tempograph.db`. Tables: `frames`, `detections` (+ `mask_rle`), `depth_frames`, `audio_segments`, `run_stages` (resume guards), `ethogram_labels`, `run_meta` (key-value), `frame_captions` (dense captions). Bbox coords normalised to saved JPEG dims (0..1).

**External services:**
- **llama-server** (Qwen3.5-VL): systemd `--user` unit `qwen-tempograph.service`, port **8082**. Pipeline auto-starts/stops via `systemctl`. Every request passes `chat_template_kwargs: {enable_thinking: false}`.
- **Ornith 9B walker**: default port **8085** (`TEMPOGRAPH_WALKER_URL`).
- **EscalationVerifier**: default port **8096** (`TEMPOGRAPH_VERIFIER_URL`).
- **whisper.cpp**: built at `~/whisper.cpp`, Vulkan backend, default device 1 (NVIDIA).

## Coding Conventions

- Python 3.9+, type hints on all functions, 88-char lines (Black).
- Imports: stdlib → third-party → local (`src.`). Alphabetical within groups.
- Naming: Classes=PascalCase, functions=snake_case, constants=SCREAMING_SNAKE_CASE.
- Models: Pydantic v2 in `src/models.py`.
- Error handling: explicit types, log before raising, handle GPU/CUDA gracefully.
- GPU cleanup: `gc.collect()` + `torch.cuda.empty_cache()` after model use.
- Docstrings: Google format, document exceptions.
- Tests: `pytest` in `tests/`, file `test_*.py`, class `Test*`. `conftest.py` adds project root to `sys.path`.
- `.flake8`: `max-line-length = 88`, `extend-ignore = E203`.

## VRAM Budget

Pipeline fits within 6 GB VRAM via sequential loading:

1. Frame extraction (CPU)
2. YOLO detection (~0.3 GB)
3. Depth estimation (~0.5 GB) — opt-in
4. Unload vision models
5. Dense captioning (9B VLM, ~0.5 GB) — opt-in
6. VLM captioning (~2 GB Q4) — external llama-server
7. Unload VLM
8. Whisper audio (CPU or Vulkan ~300 MB)
9. Aggregation (CPU)

When adding new models, ensure they fit or add a VRAM verification test.

## Footguns

1. **`depth-anything-v2` PyPI**: `depth-anything-v2>=1.0.0` is unsatisfiable (only 0.1.0 exists). Pipeline uses `transformers.pipeline("depth-estimation", ...)` directly.
2. **AMD radv `vk::DeviceLostError`**: Whisper defaults to Vulkan device 1 (NVIDIA). Falls back to NVIDIA → CPU.
3. **Filename = output dir**: same filename overwrites previous run results.
4. **4 GB Streamlit upload**: `uploaded.read()` holds entire file in RAM. Bootstrap creates `.streamlit/config.toml` with cap.
5. **Mask persistence**: `mask_rle` column added but seg variant masks not fully wired into all UI paths.
6. **Pre-2026-04-27 bboxes**: old runs have bboxes normalised against source video dims. Re-run to fix.
7. **LLM backend URL**: hardcoded `http://127.0.0.1:8082` in summarizer/Results. Should read from config.
8. **ETA calibration**: `record_stage_timing()` exists but isn't wired into `pipeline_v2.run()`.

## GPU Work Policy

- GPU work only on **CUDA device 0** (RTX 3060) unless explicitly instructed.
- The AMD 9070 XT serves the Qwen3.5-VL model — **do not allocate memory on it**.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TEMPOGRAPH_VLM_URL` | `http://127.0.0.1:8085` | llama-server address for Qwen3.5-VL |
| `TEMPOGRAPH_VLM_MODEL` | `ornith-1.0-9b-Q4_K_M.gguf` | Model name in model directory |
| `TEMPOGRAPH_WALKER_URL` | (inherits VLM_URL) | Optional walker service |
| `TEMPOGRAPH_VERIFIER_URL` | `http://127.0.0.1:8096` | Optional verifier service |
| `TEMPOGRAPH_WHISPER_BIN` | `~/whisper.cpp/build/bin/whisper-cli` | Path to whisper binary |
| `TEMPOGRAPH_WHISPER_MODELS` | `~/whisper.cpp/models` | Whisper model directory |
| `TEMPOGRAPH_RESULTS_DIR` | `results` | Output directory |

## Where to Look Next

- `TODO.md` — next tasks, constraints, environment facts
- `QUEUE.md` — v1.0 ship plan and post-ship backlog
- `SUMMARY.md` — session-by-session change history, acceptance test outputs
- `docs/PIPELINE.md` — deep stage-by-stage internals
- `docs/HARDWARE.md` — per-stage VRAM/runtime breakdown

## Git Conventions

- Run lint/format before committing: `black src/ tests/ && flake8 src/ tests/ && pytest tests/`
- Write meaningful commit messages; one commit per logical change minimum.
