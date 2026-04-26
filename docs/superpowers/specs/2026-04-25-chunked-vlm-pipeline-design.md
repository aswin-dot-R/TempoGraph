# TempoGraph v2 — Chunked VLM Pipeline Design

**Date:** 2026-04-25
**Status:** Approved (brainstorm complete, ready for implementation plan)
**Scope:** Replace the current "all frames in one VLM call" approach with a chunk-based pipeline using llama-server, with motion-aware frame selection, decoupled YOLO/VLM FPS, and hard-wipe context management.

---

## 1. Motivation

The current pipeline (`src/pipeline.py`) extracts adaptive keyframes, runs YOLO + depth on those, then dumps every frame into one big VLM call expecting a single JSON blob back. This fails on real videos because:

1. The VLM gets too much visual context at once with no temporal grounding
2. Frame selection ignores whether the camera itself is moving
3. YOLO and VLM run at the same cadence even though they have very different cost profiles
4. No incremental output — if the VLM call fails or times out, nothing is salvageable
5. Depth runs by default but adds load for spatial reasoning that most videos don't need

Modern SOTA (NVIDIA VSS, LLMVS CVPR 2025, StreamingVLM) all converge on **chunked per-segment captioning + dedicated aggregation**. We adopt that pattern, scoped to a single local llama-server instance.

---

## 2. High-Level Architecture

```
                Video file
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 1: Frame Selection │
        │  - delta computation     │
        │  - camera-mode aware     │
        │  - UI delta plot         │
        └──────────────────────────┘
                    |
        green + orange frame set
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 2: YOLO Sweep      │  (always on)
        │  - bbox per frame → DB   │
        │  - track IDs assigned    │
        │  - compressed JPEGs      │
        └──────────────────────────┘
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 3: Depth (optional)│  (off by default)
        │  - depth map per frame   │
        │  - stored as .npy in DB  │
        └──────────────────────────┘
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 4: VLM Frame Pick  │
        │  - score each YOLO frame │
        │  - take top-K → VLM set  │
        └──────────────────────────┘
                    |
        chunks of N frames each
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 5: Chunked VLM     │
        │  - per-chunk caption     │
        │  - YOLO/depth injected   │
        │  - hard-wipe + 1-line    │
        │    seed from prev chunk  │
        └──────────────────────────┘
                    |
        per-chunk caption log
                    |
                    v
        ┌──────────────────────────┐
        │ Stage 6: Aggregation     │
        │  - LLM call: captions →  │
        │    AnalysisResult JSON   │
        │  - hierarchical if long  │
        └──────────────────────────┘
                    |
                    v
              graph + timeline
```

Audio (Whisper) runs in parallel on CPU as today. No change.

---

## 3. Stage 1: Frame Selection

### 3.1 Static camera mode (CCTV / fixed)

Reuses the existing `_scan_deltas` logic in `frame_extractor.py`.

```
1. Scan full video, compute mean abs pixel delta on 160×N grayscale thumbnails
2. Adaptive threshold = median(deltas) + 1·std(deltas)
3. Frames above threshold → KEYFRAMES (mandatory, plotted green)
4. Uniformly sample at user FPS (default 1Hz) → SAMPLED (plotted orange)
5. Final YOLO set = green ∪ orange (deduplicated, sorted by frame index)
```

### 3.2 Moving camera mode (handheld / drone / dashcam)

Pixel delta is dominated by camera motion in this case, so we compensate first:

```
1. For each consecutive pair (frame N-1, frame N):
   a. Detect ORB features on both
   b. Match features, RANSAC homography H
   c. Warp frame N-1 → aligned with frame N: prev_warped = warp(prev, H)
   d. residual_delta = mean(|frame_N - prev_warped|)
2. Apply same threshold logic to residual_delta to identify keyframes
3. Same uniform sampling at user FPS for orange frames
```

Failure modes for homography (no features, fast zoom, full occlusion) fall back to raw pixel delta with a logged warning.

### 3.3 Camera mode selector (UI)

```
Camera type:  ( ) Static / fixed (CCTV)
              (•) Moving / handheld
              ( ) Auto-detect
```

**Auto-detect heuristic:** Compute median displacement of matched ORB features across the first 30 sampled frames. If median displacement > 5% of frame width → moving. Else → static.

### 3.4 UI: delta plot

A live plot rendered in the Streamlit page **before** the user kicks off the pipeline:

- X-axis: frame index (or timestamp)
- Y-axis: delta (raw or residual depending on mode)
- Horizontal red dashed line: adaptive threshold
- Green dots: keyframes (mandatory)
- Orange dots: uniform samples
- Sliders: threshold multiplier (default 1.0σ), uniform-sample FPS (default 1Hz)
- Live update on slider change so user can preview before running

---

## 4. Stage 2: YOLO Sweep

### 4.1 Configuration

| Option | Default | UI control |
|---|---|---|
| Enable detection | on | checkbox |
| Sweep FPS (effective via Stage 1) | 1.0 Hz | slider |
| Model variant | `yolo11n.pt` (detect) | dropdown |
| Optional segmentation | off | toggle → switches to `yolo11n-seg.pt` |
| Confidence threshold | 0.5 | slider |

### 4.2 Output → DB

Each detected box stored as a row keyed by `(frame_idx, track_id)`:

```python
{
    "frame_idx": 142,
    "timestamp_ms": 4733,
    "track_id": 3,
    "class_name": "person",
    "x1": 0.32, "y1": 0.41, "x2": 0.58, "y2": 0.92,  # normalized
    "confidence": 0.91,
    "mask_path": "results/{job_id}/masks/142_3.png"  # only if seg mode
}
```

### 4.3 Storage backend

**SQLite** at `results/{job_id}/tempograph.db` with three tables:
- `frames`: frame_idx, timestamp_ms, image_path (JPEG q=80, ~50KB each), is_keyframe, delta_score
- `detections`: frame_idx, track_id, class, bbox, conf, mask_path
- `depth_frames`: frame_idx, depth_npy_path (only populated if depth enabled)

Compressed JPEGs go to `results/{job_id}/frames/frame_NNNNN.jpg` at quality=80, max width 640px.

Why SQLite: zero setup, queryable from the API, survives crashes, simple to add an indexer later. We can swap to Postgres later without touching the rest of the pipeline.

---

## 5. Stage 3: Depth Estimation (Optional)

**Default: OFF.** Only runs when user explicitly enables it.

UI label: "Depth Estimation (spatial awareness — slower)"

When enabled:
- Runs Depth Anything V2 ViT-S on the same green+orange frame set as YOLO
- Saves per-frame depth map as `.npy` at `results/{job_id}/depth/frame_NNNNN.npy`
- Records mean depth per YOLO bbox (computed by sampling the depth map within each box) and stores it on the detection row
- This per-bbox depth is what gets injected into the VLM prompt — the VLM doesn't need the full depth map

Spatial info injected into VLM:
```
person at depth 0.32 (foreground), dog at depth 0.58 (mid), tree at depth 0.91 (background)
```

When depth is OFF, those mean-depth values are simply absent from the injection.

---

## 6. Stage 4: VLM Frame Subset Selection

The VLM sees fewer frames than YOLO — typically half or fewer. Each YOLO frame gets a score:

```python
score(frame_n) = (
    α * normalized_delta(n)              # visual change (residual if moving cam)
  + β * detection_set_change(n, n-1)     # +1 per new/missing class
  + γ * track_id_churn(n, n-1)           # +1 per new/missing track_id
  + δ * mean_iou_drop(n, n-1)            # 1 - mean IoU of matched track_ids
)

if frame is a green keyframe: score = +inf  (always selected)
```

Default weights: α=1.0, β=2.0, γ=2.0, δ=0.5

Top-K frames by score → VLM frame set, where:
```
K = ceil(video_duration_seconds * vlm_fps_user)
```

UI:
| Option | Default |
|---|---|
| VLM caption FPS | 0.5 Hz |
| Frames per chunk | 10 |

A 60s video at 0.5 Hz = 30 frames sent to VLM, in 3 chunks of 10.

---

## 7. Stage 5: Chunked VLM Captioning

### 7.1 Chunking

VLM frame set divided into consecutive chunks of N frames (N=10 by default). The last chunk may be shorter.

### 7.2 Per-chunk prompt template

```
You are watching a 10-second segment of a video. Describe what is
happening across these frames in chronological order.

Previous segment summary: {one_line_seed_or_"this is the start"}

For each frame, output ONE LINE describing the action. If consecutive
frames show no significant change, you may write "(no change)".
End with ONE LINE summarizing this segment in <= 20 words for use as
context in the next segment.

Frame data:
[frame 1 — t=00:00] YOLO: person at [0.32,0.41,0.58,0.92] conf=0.91; dog at [0.71,0.62,0.88,0.94] conf=0.85
[frame 2 — t=00:01] YOLO: person at [0.34,0.41,0.60,0.92] conf=0.92; dog at [0.69,0.61,0.86,0.93] conf=0.84
...

Output format:
FRAME_1: <description>
FRAME_2: <description>
...
SUMMARY: <one-line segment summary>
```

When depth is enabled, each YOLO line includes `depth=0.32` per object.

### 7.3 Hard-wipe context strategy

```
seed = "this is the start"
for chunk in chunks:
    response = ollama.chat(prompt(chunk, seed))
    captions[chunk.id] = parse_per_frame_lines(response)
    seed = parse_summary_line(response)   # extract just the SUMMARY: line
    # everything else is discarded — next call starts fresh
```

Token math per call (qwen3-vl:4b, 32k context):

| Component | Tokens |
|---|---|
| 10 frames (vision) | 2,500 – 10,000 |
| YOLO + depth text | 300 – 600 |
| Previous one-line seed | 30 – 50 |
| Prompt scaffold | 200 |
| **Total** | **~3k – 11k** |

Comfortably within budget. No accumulation across calls.

### 7.4 Failure handling

- If a chunk call times out or returns garbage, log warning, set `caption=None` for those frames, set `seed=` last successful seed, continue with next chunk
- Pipeline never aborts mid-video on a single bad chunk — partial results are always produced

---

## 8. Stage 6: Aggregation

After all chunks complete, we have a per-frame caption log. A second llama-server call (text-only, no images) consolidates this into the structured `AnalysisResult` JSON that the rest of the system expects.

### 8.1 Single-pass aggregation (short video, <30 chunks)

All captions fit in 32k context. One call:

```
You are given a chronological log of per-frame descriptions of a video.
Identify the entities (people, animals, vehicles, objects), their
behaviors, and produce structured JSON matching this schema: {AnalysisResult}

Caption log:
[00:00] FRAME_1: A person enters from the left
[00:01] FRAME_2: The person walks toward a desk
...

Output ONLY the JSON.
```

### 8.2 Hierarchical aggregation (long video, >30 chunks)

```
1. Group every 10 chunk-captions → ask LLM to compress into a meta-caption
2. Now you have meta-captions; if they fit in 32k, run single-pass aggregation
3. Else recurse (rare — would need a multi-hour video at high FPS)
```

The aggregation output goes through the existing `JSONParser` (lenient parser already handles malformed VLM JSON) and produces an `AnalysisResult`.

### 8.3 Graph build

Unchanged from today. `GraphBuilder` consumes the `AnalysisResult` and produces NetworkX graph + JSON + pyvis HTML.

---

## 9. Configuration & UI

### 9.1 Streamlit configuration panel

```
Pipeline Configuration
─────────────────────────────────────
Camera type
  ( ) Static / fixed   (•) Moving / handheld   ( ) Auto-detect

Object Detection (YOLO)
  ☑ Enable
  Sweep FPS:           [ 1.0 ] Hz
  Model:               [ yolo11n  ▾ ]   ( ) Use segmentation variant
  Confidence:          [ 0.50 ]

Depth Estimation
  ☐ Enable (spatial awareness — slower)

VLM Captioning (llama-server)
  ☑ Enable
  Caption FPS:         [ 0.5 ] Hz
  Frames per chunk:    [ 10 ]
  Model:               [ qwen3-vl:4b ▾ ]

Audio (Whisper)
  ☑ Enable
  Model:               [ small ▾ ]
─────────────────────────────────────
[ Preview frame selection ]   ← runs Stage 1 only, shows delta plot
[ Run full pipeline ]
```

### 9.2 Always-on backend

`backend = "llama-server"` is hard-coded in v2. The legacy `gemini` and `qwen` backends remain in the codebase for now but are not selectable from the UI. (Removing them is out of scope for this redesign.)

---

## 10. Component Boundaries & Files

| Component | File | Responsibility |
|---|---|---|
| Frame selection (delta + moving-cam compensation) | `src/modules/frame_extractor.py` | New `select_frames(camera_mode, fps, threshold_mult)` returning frame indices + delta array for plotting |
| Storage layer | `src/storage.py` *(new)* | SQLite schema + CRUD for frames, detections, depth |
| YOLO sweep | `src/modules/detector.py` | Add `detect_to_db(frames, db)` method |
| Depth | `src/modules/depth.py` | Add `estimate_to_db(frames, db)` method; per-bbox depth helper |
| VLM frame scoring | `src/modules/frame_scorer.py` *(new)* | Pure-function scorer over DB rows, returns top-K frame indices |
| Chunked VLM | `src/backends/llama_server_backend.py` | New `caption_chunks(chunks, db)` method; old `analyze_video` kept for legacy |
| Aggregation | `src/aggregator.py` *(new)* | Captions → `AnalysisResult` via second llama-server call |
| Pipeline orchestrator | `src/pipeline.py` | Sequential calls into stages 1-6 |
| UI delta preview | `ui/app.py` | Plotly chart, Stage-1-only preview button |

The existing `graph_builder.py`, `video_annotator.py`, `json_parser.py`, `models.py` need no changes — they consume the same `AnalysisResult` shape.

---

## 11. Error Handling & Robustness

| Failure | Behavior |
|---|---|
| Ollama unreachable | Fail fast with clear message before Stage 1 |
| ORB finds no features (moving mode) | Log warning, fall back to raw delta for that pair |
| YOLO finds zero detections in a frame | Frame still recorded in DB with empty detection list; VLM still gets it |
| Depth model OOM | Log warning, disable depth for the rest of the run, continue |
| Single chunk VLM call fails | Mark that chunk's captions as `None`, keep prior seed, continue |
| Aggregation call fails | Save raw caption log as `captions.json` so user has something; produce empty `AnalysisResult` with summary="aggregation failed, see captions.json" |

---

## 12. Testing

| Test | What it verifies |
|---|---|
| `test_frame_selection_static.py` | Delta thresholding picks expected keyframes on a synthetic CCTV-like clip |
| `test_frame_selection_moving.py` | Motion-compensated delta cancels out a pure pan, picks no spurious keyframes |
| `test_storage.py` | SQLite schema, CRUD, foreign-key relations |
| `test_frame_scorer.py` | Top-K selection prefers frames with detection-set changes over pure pixel delta |
| `test_chunked_vlm_mock.py` | With a mocked ollama response, verifies chunking, seed propagation, and parsing |
| `test_aggregator_mock.py` | Single-pass + hierarchical paths produce valid `AnalysisResult` |

Existing `test_parser.py` and `test_vram_budget.py` continue to apply.

---

## 13. Out of Scope (deferred)

- Replacing Gemini / Qwen local backends (kept in code, not exposed in UI)
- Real-time / streaming inference (current scope is offline batch on a complete video file)
- Multi-camera fusion
- Audio-visual correlation in chunk prompts (audio is still aggregated separately as today)
- Fine-tuning the VLM on domain-specific behavior taxonomies
- Replacing SQLite with Postgres
- Per-frame mask injection for the VLM (segmentation toggle stores masks but doesn't inject them)

---

## 14. Open Questions for Implementation Plan

These are questions to resolve when writing the implementation plan, not blockers on this design:

1. Exact JPEG quality / resolution defaults for compressed frame storage (start at q=80, w=640)
2. Whether to use `pyvis` for the delta plot or `plotly` (lean plotly — already a dependency)
3. Default ORB feature count for homography (start at 500)
4. Whether the SQLite DB lives under `results/{job_id}/` or in a global path (start with per-job for clean isolation)
