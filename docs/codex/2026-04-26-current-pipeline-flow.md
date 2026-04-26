# TempoGraph v2 Current Pipeline Flow

This document describes the latest pipeline flow implemented in
`src/pipeline_v2.py` as it exists right now, including the intended stages and
the current rough edges observed during smoke-test work.

## Overview

The latest pipeline is `PipelineV2`. It is the newest local-first pipeline in
the repo and is currently used by the Streamlit UI.

Primary entrypoints:

- CLI: `python -m src.pipeline_v2`
- Streamlit UI: `streamlit run ui/app.py`

The FastAPI API still uses the older `src/pipeline.py` path.

## Inputs

`PipelineV2` takes:

- `video_path`
- `output_dir`
- `camera_mode`: `static`, `moving`, `auto`
- `yolo_fps`
- `vlm_fps`
- `chunk_size`
- `depth_enabled`
- `use_segmentation`
- `threshold_mult`
- `skip_vlm`

It creates a SQLite database at:

- `<output>/tempograph.db`

## Stage 1: Frame Selection

Module:

- `src/modules/frame_selector.py`

What it does:

1. Opens the input video with OpenCV
2. Reads metadata such as FPS, frame count, width, and height
3. Chooses frame-selection logic based on `camera_mode`

Modes:

- `static`: grayscale thumbnail pixel-delta scan
- `moving`: motion-compensated delta using ORB feature matching and homography
- `auto`: estimates camera motion first, then picks `static` or `moving`

Output:

- `FrameSelectionResult`

Important fields:

- `frame_indices`
- `keyframe_indices`
- `sampled_indices`
- `scan_indices`
- `deltas`
- `threshold`
- `camera_mode`

Current observed state:

- `static` mode appears to progress normally
- `moving` mode is currently the weakest runtime path and, in live testing, has
  stalled before any frame rows were written

## Stage 1.5: Frame Export and Frame Rows

Still inside `src/pipeline_v2.py`.

What it does:

1. Seeks the selected frame indices in the source video
2. Writes JPEG frames into `<output>/frames/`
3. Computes timestamp milliseconds
4. Inserts rows into the SQLite `frames` table

`frames` table columns:

- `frame_idx`
- `timestamp_ms`
- `image_path`
- `is_keyframe`
- `delta_score`

This is the first durable output of the pipeline.

## Stage 2: YOLO Sweep to SQLite

Module:

- `src/modules/detector.py`

Method:

- `detect_to_db(...)`

What it does:

1. Loads an Ultralytics YOLO model
2. Uses:
   - `yolo11n.pt` normally
   - `yolo11n-seg.pt` when segmentation is enabled
3. Selects device:
   - CUDA if `torch` exists and `torch.cuda.is_available()`
   - otherwise CPU
4. Runs tracking on each exported frame
5. Normalizes bounding boxes by frame size
6. Inserts detections into SQLite

`detections` table columns:

- `detection_id`
- `frame_idx`
- `track_id`
- `class_name`
- `x1`
- `y1`
- `x2`
- `y2`
- `confidence`
- `mean_depth`

Notes:

- Detection row count may legitimately be zero on synthetic content
- The table should still exist even when nothing is detected

Environment sensitivity:

- This stage depends heavily on the active Python environment
- In the wrong environment (`base`), `ultralytics` was missing
- In the working environment (`msd`), `torch`, CUDA, and `ultralytics` exist

## Stage 3: Optional Depth to SQLite

Module:

- `src/modules/depth.py`

Method:

- `estimate_to_db(...)`

What it does:

1. Loads Depth Anything V2
2. Runs depth inference on each selected frame
3. Writes normalized `.npy` depth maps under `<output>/depth/`
4. Inserts frame rows into `depth_frames`
5. Computes per-bounding-box mean depth and updates `detections.mean_depth`

`depth_frames` table columns:

- `frame_idx`
- `depth_npy_path`

Notes:

- Only runs when `--depth` is enabled
- Depends on both model availability and successful completion of earlier stages

## Stage 4: Frame Scoring for VLM

Module:

- `src/modules/frame_scorer.py`

What it does:

1. Reads frame/detection state from SQLite
2. Scores frames for downstream VLM usefulness
3. Preserves keyframes
4. Picks top-K according to video duration and `vlm_fps`

Output:

- ordered frame indices for VLM use

This is the end of the fully local pre-VLM pipeline.

## Offline Stop Point: `--skip-vlm`

This flag was added specifically for offline testing.

Behavior:

- The pipeline stops after Stage 4
- It skips:
  - Stage 5 chunked VLM
  - Stage 6 aggregation
  - graph build
  - `analysis.json` write
- It returns `PipelineResult` with:
  - `analysis=None`
  - `processing_time` set

This is the current safest smoke-test path when llama-server is unavailable.

## Stage 5: Chunked VLM Captioning

Module:

- `src/backends/llama_server_backend.py`

Method:

- `caption_chunks(...)`

Intended flow:

1. Divide selected VLM frames into chunks
2. Build prompt text using frame timestamps and YOLO rows
3. Attach images for each chunk
4. Call an Ollama-compatible HTTP endpoint
5. Parse per-frame lines plus a chunk summary
6. Carry the summary into the next chunk as seed context

Current limitation:

- Not runnable in this engagement because no llama-server / Ollama is available

## Stage 6: Aggregation

Module:

- `src/aggregator.py`

Method:

- `aggregate(...)`

Intended flow:

1. Gather chunk captions
2. Optionally compress hierarchically for long runs
3. Make a text-only LLM call
4. Parse the final structured result into `AnalysisResult`

Expected output:

- entities
- visual events
- audio events
- multimodal correlations
- summary

Current limitation:

- Also blocked in this engagement because it depends on the same llama-server
  path

## Final Graph and Persisted Outputs

If Stages 5 and 6 succeed:

1. `GraphBuilder` builds a graph from `AnalysisResult`
2. `graph.html` is attempted with `pyvis`
3. `analysis.json` is written

Current v2 outputs are narrower than legacy outputs.

Typical current v2 outputs:

- `analysis.json` when VLM stages run
- `tempograph.db`
- `frames/*.jpg`
- optional `depth/*.npy`
- optional `graph.html`

Not currently mirrored from legacy:

- `graph.json`
- `timeline.json`
- annotated video
- integrated audio flow

## Intended End-to-End Shape

Conceptually, the v2 flow is:

`video -> frame selection -> JPEGs + SQLite -> YOLO -> optional depth -> frame scoring -> chunked VLM -> aggregation -> graph/artifacts`

That architecture is already visible in:

- `src/pipeline_v2.py`
- `src/storage.py`
- `src/modules/frame_selector.py`
- `src/modules/frame_scorer.py`
- `src/backends/llama_server_backend.py`
- `src/aggregator.py`

## Current Practical Status

Working in principle:

- static frame selection
- frame export
- SQLite setup
- offline stop point with `--skip-vlm`
- v2 UI wiring

Working only in the correct environment:

- YOLO
- CUDA path
- depth path

Unavailable in this engagement:

- chunked VLM
- aggregation
- anything requiring live llama-server

Known runtime rough edges:

- `moving` camera mode appears to stall before frame rows are written
- active environment selection matters a lot
- `base` was the wrong env for smoke testing
- `msd` is the working ML environment discovered during testing

## Practical Summary

If everything works as intended, the latest pipeline should behave like this:

`video -> selected frames -> DB + JPEGs -> detections -> optional depth -> scored VLM frames -> captions -> AnalysisResult -> graph`

In the current state of the repo, the most reliable subset is:

`video -> static frame selection -> JPEGs + SQLite -> YOLO -> optional depth -> frame scoring -> stop with --skip-vlm`
