# PS5b — Click-to-play: test suite (9B lane)

You are writing the TEST SUITE for the click-to-play feature. The
implementation is being written **in parallel by another model** against
the same frozen interfaces (`docs/ps/ps5a.md`). Code against the
contracts below — do NOT read or wait for the implementation files; they
may not exist yet. Your tests are expected to FAIL until the
implementation lands; that is normal. Your gate is: the test file is
importable, complete, and each test would pass against a correct
implementation.

## Scope fence

- **Files you may create:** `tests/test_click_to_play.py`
- Touch NOTHING else. Do NOT create or edit `ui/video_player.py` or
  `ui/pages/Results.py`.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN INTERFACES you are testing

```python
from ui.video_player import VideoSource, resolve_video, make_strip_mapper, render_player
# VideoSource: dataclass(path: Path, kind: str, time_mapper: Callable[[float], float])
# resolve_video(run_dir, db) -> Optional[VideoSource]
#   order: run_meta "video_path" if file exists (kind="original", identity mapper)
#          -> annotated_strip.mp4/.avi (kind="strip", proportional mapper)
#          -> None. Never raises.
# make_strip_mapper(source_dur_s, strip_dur_s) -> mapper
#   t * (strip_dur_s / source_dur_s), clamped to [0, strip_dur_s];
#   source_dur_s <= 0 -> always 0.0
```

Results.py session-state contract: clicking a ▶ sets
`st.session_state["player_start_s"]` (float seconds) and
`player_requested=True`. Button keys: `play_seg_{segment_id}`,
`play_cap_{frame_idx}`, `play_ev_{ev_idx}`, `play_frame_{frame_idx}`.
Transcript paginates past 200 segments in pages of 100 behind a
"Show more" button.

## Fixture recipe (copy the established patterns)

- DB fixtures: build with `TempoGraphDB` like `tests/test_dense_walker.py`
  does (`insert_frame`, `insert_audio_segment`); tiny JPEGs via PIL.
- Tiny real mp4 for the "original video" case: generate with cv2
  VideoWriter (see `tests/test_clip_export.py` for the pattern), write its
  path into `run_meta` via `db.set_meta("video_path", str(p))`.
- AppTest fixtures: copy the `TEMPOGRAPH_RESULTS_DIR` monkeypatch pattern
  from `tests/test_results_apptest.py` (fixture results dir with one run
  containing tempograph.db + frames).

## Tests to write

`tests/test_click_to_play.py`:

1. **`make_strip_mapper` math** — (source 100s, strip 25s): 0→0, 40→10,
   100→25, 200→clamped 25, negative→0; source_dur 0 → always 0.
2. **`resolve_video` order** — original present → kind="original" and
   identity mapping (mapper(7.3) == 7.3); video_path meta set but file
   deleted → falls back to strip when annotated_strip.mp4 exists; neither
   → None; no run_meta key at all → strip or None, never an exception.
3. **Transcript ▶ click** (AppTest) — fixture DB with 5 audio segments +
   original video in run_meta: page renders without exception; at least 5
   buttons whose keys start with `play_seg_`; simulating a click on
   segment 2's button (`at.button(key=...).click().run()`) sets
   `at.session_state["player_start_s"]` to that segment's start seconds.
4. **Frame inspector ▶** (AppTest) — clicking `play_frame_{idx}` sets
   `player_start_s` to `timestamp_ms/1000`.
5. **Pagination** (AppTest) — fixture with 250 segments: initial render
   shows exactly 100 `play_seg_` buttons; after clicking "Show more", 200.
6. **No-video honesty** (AppTest) — fixture with neither original nor
   strip: page renders, no exception, and no `play_seg_` click crashes it.

Mark the AppTest tests with the same style/imports as
`tests/test_results_apptest.py`. Keep each test independent (fresh
tmp_path fixtures).

## ACCEPTANCE (run, paste output)

```bash
# Import/collection gate — file must collect cleanly even before the
# implementation exists (collection errors on missing ui.video_player are
# expected ONLY at import of that module; guard with
# pytest.importorskip("ui.video_player") at the top of the file):
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_click_to_play.py --collect-only -q
/home/ashie/anaconda3/bin/python3 -m black tests/test_click_to_play.py
git status --porcelain     # only tests/test_click_to_play.py
```

## Commit message

(do not commit — the gate reviewer commits both lanes together after the
combined suite runs green)
