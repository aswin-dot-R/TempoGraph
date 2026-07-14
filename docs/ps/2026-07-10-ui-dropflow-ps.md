# Problem Statement: TempoGraph "Drop → Watch → Explore" UI + logic hardening

You are working unattended. Your finish line is the ACCEPTANCE section — run
those commands yourself; the task is done only when they all pass. Read
`docs/superpowers/specs/2026-07-10-ui-dropflow-design.md` first — it is the
approved design and this PS is its execution contract.

## Environment facts (do not rediscover)

- You are in a git worktree on branch `ui-v3-dropflow`. Commit here freely,
  small commits per component. NEVER push, never switch branches, never
  touch `~/TempoGraph` (the user's checkout with uncommitted WIP).
- Python: `/home/ashie/anaconda3/bin/python3` — has streamlit 1.51, pytest,
  torch, ultralytics. Run tests as
  `/home/ashie/anaconda3/bin/python3 -m pytest`.
- 29 existing tests exist and currently pass. Keep them passing.
- FORBIDDEN: `systemctl`, starting/stopping any llama-server, `sudo`,
  installing system packages, anything that allocates >1 GB of GPU memory
  (the AMD GPU is serving the model that is running YOU; TempoGraph's
  qwen-autostart path must stay untested — mock it).
- `ffmpeg` and `ffprobe` are on PATH. YOLO weights are in the repo root.
- The UI entry point is `ui/app.py` (750 lines); results pages live in
  `ui/pages/`. Core pipeline: `src/pipeline_v2.py`. Read `TODO.md` — it has
  exact file:line targets for Part B.

## Part A — UI redesign (primary)

Implement the three-screen flow from the design doc:

1. `src/auto_profile.py` (NEW): `probe(path) -> VideoFacts` (ffprobe JSON:
   duration_s, width, height, fps, has_audio) and
   `derive_plan(facts) -> PipelineConfig` implementing the design-doc rules.
   Pure functions, fully unit-tested with fixture probe dicts (no real
   video needed for unit tests).
2. Rewrite `ui/app.py` landing: drop zone + path/URL input + recent-runs
   gallery (read existing run DBs from `results/`). Zero sidebar widgets
   before a video is chosen.
3. Plan screen: one plan sentence (use `runtime_estimator` for the ETA) +
   **Analyze** button + collapsed "Adjust plan" expander containing the
   existing knobs pre-filled from the derived plan. The expander's values,
   if touched, override the plan.
4. Progress screen: stage checklist with status + per-stage ETA + live
   counters + **Cancel** button (needs Part B item 1).
5. Results dashboard tabs: Overview (stat cards + narrative summary —
   implement `src/summarizer.py` with an injectable LLM callable; in tests
   inject a fake; in production it calls the configured llama backend),
   Timeline, Graph (reuse existing), Ask (SQL-grounded retrieval over the
   run DB feeding the same injectable LLM), Export (wire
   `src/dataset_exporter.py`, TODO.md Tier-1).
6. Graceful stage failure: banner + continue, never a traceback in the UI.

## Part B — logic hardening (only after Part A acceptance passes)

In this order, each with tests, exact targets in `TODO.md`:
1. Cancel: `threading.Event` through `PipelineV2.run()`, checked between
   stages and inside chunk loops.
2. ETA calibration: append `(stage, elapsed_s, n_units)` to
   `results/.calibration.jsonl` after each stage; estimator prefers a
   linear fit over the last N entries, falls back to current constants.
3. Per-frame transcript join in the Frame inspector (SQL in
   `docs/PIPELINE.md`).
4. Resume-from-DB: stage guards in `pipeline_v2.run()` skipping stages
   whose outputs already exist.
5. STRETCH (skip unless everything above is green with time to spare):
   mask RLE persistence.

## ACCEPTANCE — run these; all must pass

```bash
cd /home/ashie/TempoGraph-ui-v3
PY=/home/ashie/anaconda3/bin/python3

# 1. Whole test suite (existing 29 + your new tests) green:
$PY -m pytest -x -q

# 2. AppTest UI contract (write tests/test_ui_dropflow.py):
#    - landing: exactly 0 sidebar widgets, 1 file_uploader present
#    - plan screen (simulated facts): Analyze button + Adjust expander exist
#    - expander contains the legacy knobs pre-filled from derive_plan()
$PY -m pytest tests/test_ui_dropflow.py -q

# 3. auto_profile unit tests: fixtures for (a) 90s 1080p with audio,
#    (b) 40min 720p with audio, (c) 2min silent → assert derived plans
$PY -m pytest tests/test_auto_profile.py -q

# 4. CPU end-to-end smoke (~2 min): synthetic clip, minimal profile
ffmpeg -y -f lavfi -i testsrc=duration=5:size=640x360:rate=10 /tmp/tg_smoke.mp4
$PY scripts/smoke_dropflow.py /tmp/tg_smoke.mp4   # you write this script:
#    runs pipeline with plan derived from the clip but forced:
#    device=cpu, vlm=skip, depth=skip, audio=skip
#    asserts: exit 0, run DB exists, frames>0 rows, detections table exists

# 5. Streamlit boots headless with the new UI (10s then kill):
timeout 12 $PY -m streamlit run ui/app.py --server.headless true \
  --server.port 8599 & sleep 8 && curl -sf localhost:8599 >/dev/null \
  && echo UI_BOOTS
```

## Deliverables

- Commits on `ui-v3-dropflow`, one per component, imperative messages.
- `SUMMARY.md` at repo root: what changed, what you verified (paste
  acceptance output), what remains / known gaps, screenshots not required.
- If you get irrecoverably stuck on a component, write the blocker into
  `SUMMARY.md`, commit what works, and move to the next component — a
  partial branch with green tests beats a broken complete one.
