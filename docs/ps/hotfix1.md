# HOTFIX 1 — real-video drop crashes: degrade gracefully, never die

**Bug (reproduced 2026-07-12):** dropping a real 1080p video crashes the
run. `derive_plan()` enables depth for real footage; the depth stage does
`from transformers import pipeline`; `transformers` is NOT installed in
the interpreter env → `ImportError: transformers is required for depth
estimation` → whole pipeline dies → UI shows "Pipeline failed". Smoke
clips never enable depth, so the 160-test suite never hits this path.

Repro (fails today at Depth estimation):
```
/home/ashie/anaconda3/bin/python3 -c "from transformers import pipeline"
# ModuleNotFoundError: No module named 'transformers'
```

**Design rule this PS enforces:** an optional stage (depth, audio, dense
captions) failing must cost the run that stage ONLY — banner + continue,
never a dead run. The plan must also not promise what the environment
cannot deliver.

## Scope fence

- **Files you may modify:** `src/auto_profile.py`, `src/pipeline_v2.py`
- **Files you may create:** `tests/test_optional_stage_resilience.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install` (do NOT try to install transformers —
the fix is graceful degradation, not the dependency), git push,
starting/stopping any llama-server. Interpreter:
`/home/ashie/anaconda3/bin/python3`.

## Task 1 — honest plan (`src/auto_profile.py`)

In `derive_plan()`, only enable depth when the module is importable:

```python
import importlib.util

depth_available = importlib.util.find_spec("transformers") is not None
```

`depth_enabled = <existing rule> and depth_available`. Add a
`notes: List[str]` field to `DerivedPlan` (default empty list, include it
in the dict the class exports) and append
`"depth off: transformers not installed"` when availability blocked it —
the Plan screen already renders the plan dict, so the user sees why.

## Task 2 — armor the optional stages (`src/pipeline_v2.py`)

Wrap the BODY of each optional stage in try/except so a failure emits
`self._stage("<name>", "error", error=str(e))` + a logger warning and
CONTINUES the run (no `mark_stage_complete` for the failed stage):

1. **Depth estimation** — currently unguarded: `ImportError` (or any
   exception from `estimate_to_db`) must not propagate. On failure the
   run proceeds without depth; downstream code already tolerates missing
   depth rows (bbox `mean_depth` stays NULL).
2. **Dense captions** — wrap the `run_dense_captioning(...)` call the
   same way (an unreachable walker/verifier server must not kill a run;
   today `requests` errors inside are handled per-frame, but constructor
   or DB errors are not).
3. **Audio transcription** — already has a try/except; verify it covers
   the `WhisperCppTranscriber` CONSTRUCTOR too (binary path check), not
   just `transcribe_video`. Extend if not.

Pattern (match the existing stage style exactly):

```python
try:
    <existing stage body>
except Exception as e:
    self.logger.warning(f"<Stage> failed, continuing without it: {e}")
    self._stage("<Stage>", "error", error=str(e))
```

Do NOT wrap required stages (Frame selection, YOLO detection, Frame
scoring) — a run without detections is not a run; those may still raise.

## Task 3 — tests (`tests/test_optional_stage_resilience.py`)

Copy the mocking approach of `tests/test_pipeline_v2.py` (mock every
model-touching stage). Cover:

1. **Depth ImportError doesn't kill the run** — monkeypatch
   `src.pipeline_v2.DepthEstimator` so `estimate_to_db` raises
   `ImportError("transformers is required for depth estimation")`; run
   with `depth_enabled=True`; pipeline returns a result, `run_stages`
   has NO "Depth estimation" row, and a stage event
   `("Depth estimation", "error", ...)` was emitted (capture via
   `on_stage`).
2. **Dense captioning failure doesn't kill the run** — monkeypatch
   `src.pipeline_v2.run_dense_captioning` to raise `RuntimeError`;
   `dense_captions=True`; same assertions for its stage.
3. **derive_plan honesty** — monkeypatch
   `importlib.util.find_spec` to return None for "transformers":
   `derive_plan(facts_1080p)` yields `depth_enabled=False` and a note;
   with find_spec returning a truthy spec, depth follows the existing
   rule (use the same 1080p facts fixture:
   `VideoFacts(duration_s=30.5, width=1920, height=1080, fps=30.0,
   has_audio=True)`).
4. **Whisper constructor failure doesn't kill the run** — monkeypatch
   `src.pipeline_v2.WhisperCppTranscriber` to raise `FileNotFoundError`
   on construction; `audio_enabled=True`; run completes, audio stage
   error event emitted.

## ACCEPTANCE (run, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_optional_stage_resilience.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q      # full suite green (160 + new)
# The original repro must now complete end-to-end (depth skipped honestly):
timeout 300 /home/ashie/anaconda3/bin/python3 -m src.pipeline_v2 \
  --video /home/ashie/Downloads/file_example_MP4_1920_18MG.mp4 \
  --output /tmp/hotfix1_repro --skip-vlm 2>&1 | tail -3
/home/ashie/anaconda3/bin/python3 -m black src/auto_profile.py src/pipeline_v2.py tests/test_optional_stage_resilience.py
git status --porcelain    # only the 3 scoped files
```

## Commit message

```
Hotfix: optional stages degrade gracefully instead of killing real-video runs
```

## Note for the human (not the agent)

The full fix for depth itself is `pip install transformers` into
`/home/ashie/anaconda3` — after that the plan re-enables depth
automatically. This PS makes the app honest and unkillable either way.
