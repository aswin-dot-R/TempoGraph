# HOTFIX 1a — graceful degradation: IMPLEMENTATION (35B lane)

Bug context and design rule: see `docs/ps/hotfix1.md` (reproduced crash:
real 1080p video → depth enabled by plan → `from transformers import
pipeline` → ImportError kills the run). The TEST SUITE for this fix is
being written **in parallel by another model** (`docs/ps/hotfix1b.md`)
against the FROZEN CONTRACTS below — implement them exactly.

## Scope fence

- **Files you may modify:** `src/auto_profile.py`, `src/pipeline_v2.py`
- Touch NOTHING else. Do NOT create or edit any test file —
  `tests/test_optional_stage_resilience.py` belongs to the other lane.

## Forbidden

`sudo`, `systemctl`, `pip install` (do NOT install transformers — the fix
is degradation, not the dependency), git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN CONTRACTS (the parallel test lane codes against these)

1. `DerivedPlan` gains field `notes: List[str]` (default `[]`), included
   in its exported dict under key `"notes"`.
2. `derive_plan()`: depth is enabled only when
   `importlib.util.find_spec("transformers") is not None` AND the
   existing rule says so. When availability alone blocks it, append
   exactly the note string `"depth off: transformers not installed"`.
3. Optional-stage failure behavior in `PipelineV2.run()` — for stages
   "Depth estimation", "Dense captions", "Audio transcription":
   any exception inside the stage body results in
   `self._stage("<stage name>", "error", error=str(e))`, a logger
   warning, NO `mark_stage_complete()` for that stage, and the run
   CONTINUES to completion. Stage names verbatim as quoted.
4. Required stages (Frame selection, YOLO detection, Frame scoring) keep
   raising on failure — do not wrap them.
5. Monkeypatch surface (do not rename/move these): the test lane patches
   `src.pipeline_v2.DepthEstimator`, `src.pipeline_v2.run_dense_captioning`,
   `src.pipeline_v2.WhisperCppTranscriber`.

## Tasks

1. **`src/auto_profile.py`** — contracts 1–2. Keep the module pure
   (no torch imports; `importlib.util` only).
2. **`src/pipeline_v2.py`** — contract 3, following the existing stage
   style. For Audio transcription: the existing try/except must also
   cover the `WhisperCppTranscriber(...)` constructor call (it may
   raise FileNotFoundError on a missing binary) — move it inside if
   needed. For Dense captions: wrap the whole `run_dense_captioning`
   call block. For Depth: wrap the estimator construction + 
   `estimate_to_db` block.

## ACCEPTANCE (run, paste output — new tests land in the other lane)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q      # existing 160 stay green
# The original crash repro must now complete end-to-end:
timeout 300 /home/ashie/anaconda3/bin/python3 -m src.pipeline_v2 \
  --video /home/ashie/Downloads/file_example_MP4_1920_18MG.mp4 \
  --output /tmp/hotfix1_repro --skip-vlm 2>&1 | tail -3
/home/ashie/anaconda3/bin/python3 -c "
from src.auto_profile import probe, derive_plan
p = derive_plan(probe('/home/ashie/Downloads/file_example_MP4_1920_18MG.mp4'))
print('depth_enabled:', p.depth_enabled); print('notes:', p.notes)"
/home/ashie/anaconda3/bin/python3 -m black src/auto_profile.py src/pipeline_v2.py
git status --porcelain    # only the 2 scoped files
```

## Commit message

(do not commit — the gate reviewer commits both lanes together)
