# Codex Task Brief — 2026-04-26

You are Codex. Claude orchestrates; you execute the heavy work below so we don't burn Claude tokens. Repo: `/home/ashie/TempoGraph`. Branch: `v2-chunked-pipeline`. Do not merge to main; commit on this branch.

## Hard Constraints

1. **No llama-server / Ollama is available right now.** Do not attempt to call `LlamaServerBackend.caption_chunks` against a live server. Stages 5 (chunked VLM) and 6 (aggregation) are **skipped** for this round of testing. Mock or stop short.
2. **GPU may or may not be present.** Code already has soft torch guards. Detect and report; don't fail the whole run if CUDA is missing — fall back to CPU.
3. **Don't touch `main`.** Work on `v2-chunked-pipeline`. Author identity: `aswin <aswinrajan2002@gmail.com>`. Use `git -c user.email=... -c user.name=... commit` if your global config isn't set.
4. **Read before editing.** The pipeline already exists. Don't rewrite stages — just add the flags / script / docs needed to make them testable end-to-end.

## Background

`PipelineV2` (`src/pipeline_v2.py`) runs six stages. Stages 1–4 are local (CV + ML), Stages 5–6 require llama-server:

| # | Stage | Module | Needs llama-server? |
|---|---|---|---|
| 1 | Frame selection (delta / motion-comp) | `src/modules/frame_selector.py` | No |
| 2 | YOLO sweep → SQLite | `src/modules/detector.py` (`detect_to_db`) | No |
| 3 | Depth (optional) → SQLite + .npy | `src/modules/depth.py` (`estimate_to_db`) | No |
| 4 | Frame scoring → top-K for VLM | `src/modules/frame_scorer.py` | No |
| 5 | Chunked VLM captioning | `src/backends/llama_server_backend.py` | **Yes — skip** |
| 6 | Aggregation (single-pass / hierarchical) | `src/aggregator.py` | **Yes — skip** |

Storage: `src/storage.py::TempoGraphDB` (SQLite at `<output>/tempograph.db` with tables `frames`, `detections`, `depth_frames`).

CLI today: `python -m src.pipeline_v2 --video ... --output ... --camera static [--depth] [--seg] ...`. There is no flag to stop before VLM yet — Task A adds it.

---

## Tasks

### Task A — Add `--skip-vlm` flag to the v2 CLI  *(small, do first)*

Edit `src/pipeline_v2.py`:

1. Add `argparse` flag `--skip-vlm` (action="store_true").
2. Add constructor kwarg `skip_vlm: bool = False` on `PipelineV2`.
3. In `run()`, if `self.skip_vlm`: after Stage 4 finishes, skip Stages 5, 6, graph build, and `analysis.json` write. Return a `PipelineResult` with `analysis=None` (or a stub `AnalysisResult` with empty lists if the pydantic model rejects None — check `src/models.py`). `processing_time` should still be set. Log `"Skipping VLM stages (skip_vlm=True). Stopped after frame scoring."`.
4. Wire `args.skip_vlm` through to the constructor in `__main__`.

Commit message: `v2: add --skip-vlm flag for offline pipeline testing`.

### Task B — Generate a synthetic test video  *(no external downloads)*

Create `tools/make_test_video.py`. It writes a 10-second 480p mp4 to `tests/fixtures/sample.mp4` using OpenCV's `VideoWriter`. The video must:

- Be 30 fps, 640×480.
- Contain at least 3 visually distinct "scenes" so the delta plot has real keyframes (e.g., 0–3s solid color A with a moving rectangle, 3–6s color B with a circle bouncing, 6–10s color C with two moving shapes).
- Have moments where YOLO will likely detect something — paste a small picture of a person/car or just rely on shapes (YOLO will mostly find nothing, that's fine, the test just needs the *pipeline* to run; report 0-detection frames as a known limitation).

If `tests/fixtures/sample.mp4` already exists, exit 0 without overwriting.

Add `tests/fixtures/` to `.gitignore` if the mp4 is large (>2 MB); otherwise commit the fixture so CI doesn't have to regenerate it. Use your judgment.

Commit message: `tools: synthetic test video generator for v2 smoke tests`.

### Task C — End-to-end smoke test  *(the main deliverable)*

Create `scripts/smoke_test_v2.py`. It should:

1. Ensure `tests/fixtures/sample.mp4` exists; if not, call `tools/make_test_video.py` to make it.
2. Run the v2 pipeline with `--skip-vlm` and **all defaults** otherwise: `python -m src.pipeline_v2 --video tests/fixtures/sample.mp4 --output results/smoke_test --skip-vlm`.
3. After the run, **independently verify** outputs (don't trust the process exit code alone):
   - `results/smoke_test/tempograph.db` exists.
   - Open it with `sqlite3` and assert: `frames` table has ≥1 row, `detections` table exists (rows may be 0 if YOLO finds nothing — that's OK, log it), `depth_frames` is empty (depth not enabled by default).
   - `results/smoke_test/frames/` contains JPEGs equal in count to the `frames` row count.
   - No `analysis.json` (because `--skip-vlm`).
4. Run a second pass with `--depth --camera static` and verify `depth_frames` is now populated and `results/smoke_test_depth/depth/*.npy` files exist.
5. Run a third pass with `--camera moving` (re-uses the same fixture; the moving-camera path uses ORB+RANSAC on whatever motion is in the video) and verify it doesn't crash and produces frames.
6. Print a `PASS`/`FAIL` summary table per pass with timing and row counts. Exit 1 if any assertion fails.

The script must be runnable as `python scripts/smoke_test_v2.py` from repo root. No pytest, no extra dependencies beyond what's already in `requirements.txt`.

Commit message: `scripts: end-to-end smoke test for v2 pipeline (skip-vlm)`.

### Task D — Run the smoke test and produce a report

Run `python scripts/smoke_test_v2.py` and capture stdout+stderr to `docs/codex/2026-04-26-smoke-test-report.md`. The report should include:

- Environment (python version, torch/cuda availability, ffmpeg version, OS).
- Per-pass results table (pass name, status, elapsed, frames written, detections, depth rows).
- Any errors / warnings observed.
- Whether YOLO actually detected anything in the synthetic video (and why if 0 — synthetic shapes likely won't match COCO classes, that's expected).
- Anything you had to fix or work around (link to commit SHAs).

If anything fails: **stop, fix the underlying bug in the pipeline (not the test), commit the fix, re-run**. Don't paper over real failures.

Commit message: `docs: smoke test report for v2 pipeline (2026-04-26)`.

### Task E — Update `README.md` for v2  *(documentation)*

The current `README.md` describes v1 (adaptive keyframe extraction + Gemini/Qwen choice, `python -m src.pipeline …`). Rewrite the relevant sections to reflect v2:

- Pipeline diagram updated to show: frame selection (static/moving) → YOLO+SQLite → optional depth → frame scoring → chunked VLM (hard-wipe) → aggregation.
- Quick-start uses `python -m src.pipeline_v2 --video your.mp4 --output results/run --camera static` (and a `--skip-vlm` example for the no-llama path).
- Note that llama-server (Ollama with qwen3-vl) is currently the only supported VLM backend.
- Mention the Streamlit UI (`streamlit run ui/app.py`) and its camera-mode + separate FPS controls.
- Keep the v1 CLI section under a "Legacy v1 (deprecated)" subheading — don't delete, since v1 code still ships.

Don't add badges, emojis, or marketing fluff. Match the existing tone.

Commit message: `docs: update README for v2 chunked pipeline`.

### Task F — Update `AGENTS.md` module map  *(documentation)*

Add a section describing the v2 module layout (`src/storage.py`, `src/aggregator.py`, `src/pipeline_v2.py`, `src/modules/frame_selector.py`, `src/modules/frame_scorer.py`) and the new `detect_to_db`/`estimate_to_db` methods. Keep existing v1 entries.

Commit message: `docs: extend AGENTS.md with v2 module map`.

### Task G — Reconcile `requirements.txt` with actual imports  *(small, real bug)*

`requirements.txt` is missing several packages the codebase imports. A fresh `pip install -r requirements.txt` cannot start the API or render the graph HTML.

Confirmed missing (verified by Claude — `grep -n "^<pkg>" requirements.txt` returns nothing for these):

- `fastapi` — used by `src/api.py`
- `uvicorn` — used by `src/api.py` runtime
- `pyvis` — used by `src/graph_builder.py` for HTML graph export
- `requests` — used by `src/backends/llama_server_backend.py`

Add each with a sensible lower bound (match the version range Python 3.13 supports). Don't pin tight unless you've actually tested. Group them logically in the file (e.g., `requests` near other HTTP libs, `fastapi`/`uvicorn` under an "API" heading).

After adding, run `pip install -r requirements.txt --dry-run` if your environment supports it, or just `pip install` the new ones to confirm they resolve.

Commit message: `requirements: add missing fastapi, uvicorn, pyvis, requests`.

### Task H — Fix model-name inconsistencies  *(small, contains a real bug)*

There are three identifier inconsistencies in the codebase. Fix each:

1. **`src/modules/detector.py:55` — broken default model path.** The constructor default is `model_path: str = "yolo26n.pt"`. There is no YOLO 26. Anyone instantiating `ObjectDetector()` directly (library mode, tests, future callers) will fail when ultralytics tries to download a non-existent file. `pipeline_v2.py:89` happens to override this with `yolo11n.pt`, which is why nothing has crashed yet.

   Change the default to `"yolo11n.pt"` to match the rest of the codebase. Update the docstring on lines 4–10 to match (it already lists `yolo11n.pt` / `yolo11s.pt` correctly, just confirm).

2. **`src/backends/qwen_backend.py` — three different Qwen identifiers in one file.**
   - Line 10 docstring says `Qwen3-VL`.
   - Lines 24, 29, 113 default to `Qwen/Qwen2.5-VL-7B-Instruct`.
   - Lines 133, 186, 196 log messages say `Qwen2.5-VL-3B`.

   Pick one identifier — `Qwen/Qwen2.5-VL-7B-Instruct` is the actual default the code loads, so align the log messages and docstring to that. Don't change the default kwarg unless you've verified a different model fits the VRAM budget.

3. **`configs/default.yaml:9`** — comment says `# YOLOv8-nano` but the codebase uses YOLO 11. Update the comment to `# YOLO 11 nano (yolo11n.pt)`.

Don't restructure or rename the classes — only update strings (defaults, log messages, comments, docstrings) so they tell a single coherent story.

Commit message: `fix: reconcile YOLO/Qwen model identifiers across modules`.

---

## Stretch (only if A–H finish cleanly with time to spare)

- **I.** Add a `--no-graph` flag to skip the pyvis HTML build (useful for headless CI).
- **J.** Add tqdm progress bars to YOLO sweep (`detect_to_db`) and depth sweep (`estimate_to_db`). Tqdm is already a transitive dep — confirm before importing.
- **K.** Add a `PipelineResult.stage_timings: dict[str, float]` field and populate it. Update `tests/test_pipeline_v2.py`.
- **L.** Resolve the per-bbox depth normalization deviation flagged in the plan (Task 6 of `docs/superpowers/plans/2026-04-25-chunked-vlm-pipeline.md`): decide raw vs normalized, update `estimate_to_db`, and adjust the test. Document the choice in the smoke test report.

Don't start I–L until A–H are committed.

---

## Reporting Back

When done, append a final section to `docs/codex/2026-04-26-smoke-test-report.md`:

```
## Summary
- Tasks completed: A, B, C, D, E, F, G, H[, I, J, ...]
- Commits: <SHA list>
- Tests: <pytest summary — how many pass>
- Smoke test: PASS / FAIL
- Open issues for Claude to triage: <list>
```

That's the file Claude reads when you hand back.
