# TempoGraph

**Turn video into structured entities, events, and graphs.**

TempoGraph is a multimodal video analysis project with two pipeline generations
currently living side by side:

- `src/pipeline.py`: the original multi-backend pipeline
- `src/pipeline_v2.py`: the newer chunked local pipeline

This README reflects the repository as it exists now.

## Current Status

The repo is in a transition state.

- The Streamlit UI in `ui/app.py` runs `PipelineV2`.
- The FastAPI app in `src/api.py` still runs the legacy `Pipeline`.
- The legacy CLI and the v2 CLI both exist.
- Shared contracts such as `AnalysisResult`, `GraphBuilder`, and `JSONParser`
  are reused across both paths.

## Architecture Overview

### Legacy pipeline

The legacy path in `src/pipeline.py` runs a sequential multimodal pipeline:

1. Adaptive frame extraction with `src/modules/frame_extractor.py`
2. Optional YOLO detection
3. Optional depth estimation
4. VLM analysis through one of:
   - `gemini`
   - `qwen`
   - `llama-server`
5. Optional Whisper transcription merge for local `qwen`
6. Graph export and optional annotated video generation

Backends used by the legacy pipeline:

- `src/backends/gemini_backend.py`
- `src/backends/qwen_backend.py`
- `src/backends/llama_server_backend.py`

### V2 pipeline

The newer path in `src/pipeline_v2.py` is local-first and chunked:

1. Motion-aware frame selection with `src/modules/frame_selector.py`
2. JPEG frame export plus SQLite persistence with `src/storage.py`
3. YOLO sweep into the database
4. Optional depth estimation into the database
5. Top-K frame scoring with `src/modules/frame_scorer.py`
6. Chunked captioning with `src/backends/llama_server_backend.py`
7. Aggregation back into `AnalysisResult` with `src/aggregator.py`

This is the path currently exposed by the Streamlit UI.

## What Each Backend Actually Does

### Gemini

`src/backends/gemini_backend.py` uploads the source video to Gemini, waits for
the uploaded file to become active, requests JSON output, parses it through
`JSONParser`, and deletes the uploaded file afterward.

This is the only backend in the repo that handles video and audio together in a
single model call.

### Qwen

`src/backends/qwen_backend.py` runs a quantized local Qwen2.5-VL model over
extracted frames only. Audio is not handled by the backend itself and is merged
separately in the legacy pipeline through Whisper.

### Llama Server

`src/backends/llama_server_backend.py` talks to an Ollama-compatible HTTP
server. It supports both:

- the legacy single-call frame analysis path
- the v2 chunked captioning path

## Shared Components

- `src/json_parser.py`: strips `<think>` blocks, extracts embedded JSON, fixes
  common formatting issues, and returns an `AnalysisResult` rather than
  crashing on malformed model output.
- `src/graph_builder.py`: builds a `networkx.MultiDiGraph`, exports JSON,
  generates `pyvis` HTML, and exposes simple query helpers.
- `src/video_annotator.py`: overlays timestamps, detections, depth maps, and
  active event text on top of the original video.

## Outputs

### Legacy pipeline outputs

When using `src/pipeline.py`, the repository can produce:

- `analysis.json`
- `detection.json` when detection runs
- `graph.json`
- `timeline.json`
- `graph.html` when `pyvis` is available
- `annotated.mp4` when detection or depth is enabled

### V2 pipeline outputs

When using `src/pipeline_v2.py`, the repository currently produces:

- `analysis.json`
- `tempograph.db`
- extracted JPEG frames under `frames/`
- optional normalized depth maps under `depth/`
- `graph.html` when `pyvis` is available

The v2 path does not currently have full parity with the legacy path on audio,
timeline export, API wiring, or annotated video output.

## Running The Project

### Streamlit UI

The current UI is the v2 path:

```bash
streamlit run ui/app.py
```

Before using it, start Ollama and pull the model:

```bash
ollama serve
ollama pull qwen3-vl:4b
```

The UI currently exposes:

- camera mode: `static`, `moving`, `auto`
- YOLO sweep FPS and confidence
- optional segmentation toggle
- optional depth
- VLM caption FPS and chunk size
- frame-selection preview plot

It does not currently expose the older `gemini` or `qwen` backend choices.

### Legacy CLI

```bash
python -m src.pipeline --video sample.mp4 --backend gemini --output results/legacy_gemini
python -m src.pipeline --video sample.mp4 --backend qwen --modules behavior,detection,audio
python -m src.pipeline --video sample.mp4 --backend llama-server --output results/legacy_llama
```

### V2 CLI

```bash
python -m src.pipeline_v2 \
  --video sample.mp4 \
  --output results/v2_run \
  --camera auto \
  --yolo-fps 1.0 \
  --vlm-fps 0.5 \
  --chunk-size 10
```

### REST API

The API currently routes to the legacy pipeline:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@clip.mp4" \
  -F "backend=gemini" \
  -F "modules=behavior,detection,audio"
```

### Docker

`docker-compose.yml` defines:

- `tempograph-gpu`: API + Streamlit together with GPU reservation
- `tempograph-cpu`: API + Streamlit together without GPU reservation

## Configuration

`configs/default.yaml` is still closest to the legacy pipeline shape. The
current v2 UI mostly uses runtime controls instead of consuming that file
end-to-end, so treat the YAML as a legacy-oriented default config rather than
the canonical source of truth for the entire repo.

## Tests

The test suite covers both original and v2 code paths:

- `tests/test_parser.py`
- `tests/test_vram_budget.py`
- `tests/test_storage.py`
- `tests/test_frame_selector.py`
- `tests/test_frame_scorer.py`
- `tests/test_chunked_vlm.py`
- `tests/test_aggregator.py`
- `tests/test_pipeline_v2.py`

Run all tests with:

```bash
pytest tests/
```

## Repository Structure

```text
TempoGraph/
├── src/
│   ├── pipeline.py
│   ├── pipeline_v2.py
│   ├── aggregator.py
│   ├── api.py
│   ├── graph_builder.py
│   ├── json_parser.py
│   ├── models.py
│   ├── storage.py
│   ├── video_annotator.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── gemini_backend.py
│   │   ├── llama_server_backend.py
│   │   └── qwen_backend.py
│   └── modules/
│       ├── audio.py
│       ├── depth.py
│       ├── detector.py
│       ├── frame_extractor.py
│       ├── frame_scorer.py
│       └── frame_selector.py
├── ui/
│   └── app.py
├── configs/
│   └── default.yaml
├── docs/
│   └── superpowers/
├── tests/
└── docker-compose.yml
```

## Documentation Map

- `README.md`: current project overview
- `docs/superpowers/specs/2026-04-25-chunked-vlm-pipeline-design.md`:
  v2 design intent
- `docs/superpowers/plans/2026-04-25-chunked-vlm-pipeline.md`:
  v2 implementation plan
- `todo.md`, `todo_opus.md`, `OPUS_COMPLETION_SUMMARY.md`:
  historical build artifacts, not current source-of-truth docs

## Recommendations

### What can be done better

1. Unify entrypoints around one primary pipeline. Right now the UI uses
   `PipelineV2` while the API uses `Pipeline`, so behavior depends on how the
   project is launched.
2. Make configuration one-source-of-truth. `configs/default.yaml`, the legacy
   CLI, and the v2 UI currently express overlapping settings in different ways.
3. Bring `requirements.txt` in line with actual imports. The codebase imports
   packages such as `fastapi`, `uvicorn`, `pyvis`, and `requests`, but the
   manifest does not fully reflect that.
4. Decide whether legacy and v2 outputs should converge. The legacy path
   exports `graph.json`, `timeline.json`, and annotated video, while v2 focuses
   on `analysis.json` and SQLite-backed intermediates.
5. Add smoke tests for the real entrypoints. The repo has useful unit tests,
   but it would benefit from coverage of the actual CLI, API, and UI-backed
   execution paths.
6. Standardize model naming and defaults. There are still inconsistencies
   between comments, docs, and code for YOLO and Qwen model identifiers.
7. Keep archival docs clearly labeled as historical so new contributors do not
   mistake implementation plans for current architecture.

## License

MIT
