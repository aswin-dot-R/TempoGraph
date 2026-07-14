# HOTFIX 1b — graceful degradation: TEST SUITE (9B lane)

You are writing the tests for the graceful-degradation hotfix. The
implementation is being written **in parallel by another model**
(`docs/ps/hotfix1a.md`) against the same frozen contracts. Do NOT read or
wait for the implementation — code against the contracts below. Your
tests may FAIL until the implementation lands; that is expected. Your
gate: the file collects cleanly and each test is correct against the
contracts.

## Scope fence

- **Files you may create:** `tests/test_optional_stage_resilience.py`
- Touch NOTHING else. Do NOT edit `src/auto_profile.py` or
  `src/pipeline_v2.py`.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN CONTRACTS you are testing

1. `DerivedPlan.notes: List[str]` (default empty). When transformers is
   unavailable and the size rule wanted depth, `derive_plan` sets
   `depth_enabled=False` and notes contains exactly
   `"depth off: transformers not installed"`.
2. In `PipelineV2.run()`, an exception inside an optional stage —
   "Depth estimation", "Dense captions", "Audio transcription" — emits
   stage event `(name, "error", {"error": <str>})` via the `on_stage`
   callback, writes NO `run_stages` row for that stage, and the run
   still completes and returns a result.
3. Patchable names, verbatim: `src.pipeline_v2.DepthEstimator`,
   `src.pipeline_v2.run_dense_captioning`,
   `src.pipeline_v2.WhisperCppTranscriber`.
4. Pipeline kwargs you'll need: `depth_enabled=True`,
   `dense_captions=True`, `audio_enabled=True`.

## Fixture pattern (copy from `tests/test_pipeline_v2.py`)

That file builds a tiny real mp4 with cv2 (`_make_video`), constructs
`PipelineConfig(video_path=..., output_dir=...)`, and patches
`src.pipeline_v2.ObjectDetector` / `LlamaServerBackend` /
`CaptionAggregator` with MagicMocks so no model loads. Reuse that exact
approach (import or replicate `_make_video`). Capture stage events by
passing `on_stage=lambda n, e, i: events.append((n, e, i))`.

To read `run_stages` after a run:
`sqlite3.connect(out/"tempograph.db").execute("SELECT stage_name FROM run_stages").fetchall()`.

## Tests to write

`tests/test_optional_stage_resilience.py`:

1. **Depth ImportError doesn't kill the run** — patch
   `src.pipeline_v2.DepthEstimator` so `estimate_to_db` raises
   `ImportError("transformers is required for depth estimation")`
   (constructor may also raise — cover with `side_effect` on the class if
   simpler); run with `depth_enabled=True` (+ the standard detector/VLM/
   aggregator mocks); assert: result returned, `("Depth estimation",
   "error", ...)` in events, no "Depth estimation" row in `run_stages`.
2. **Dense captioning failure doesn't kill the run** — patch
   `src.pipeline_v2.run_dense_captioning` with
   `side_effect=RuntimeError("verifier down")`; `dense_captions=True`;
   same three assertions for "Dense captions".
3. **Whisper constructor failure doesn't kill the run** — patch
   `src.pipeline_v2.WhisperCppTranscriber` with
   `side_effect=FileNotFoundError("no whisper-cli")`;
   `audio_enabled=True`; same three assertions for "Audio transcription".
4. **derive_plan honesty, unavailable** — monkeypatch
   `importlib.util.find_spec` (as imported in `src.auto_profile`) to
   return None for `"transformers"`; with
   `VideoFacts(duration_s=30.5, width=1920, height=1080, fps=30.0,
   has_audio=True)` assert `depth_enabled is False` and the exact note
   string present.
5. **derive_plan honesty, available** — find_spec returns a truthy
   object → `depth_enabled` follows the pre-existing size rule for the
   same facts (assert True for this 1080p fixture) and no such note.
6. **Happy path unchanged** — all optional stages mocked to succeed →
   no "error" events at all.

## ACCEPTANCE (run, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_optional_stage_resilience.py --collect-only -q
/home/ashie/anaconda3/bin/python3 -m black tests/test_optional_stage_resilience.py
git status --porcelain    # only tests/test_optional_stage_resilience.py
```

## Commit message

(do not commit — the gate reviewer commits both lanes together)
