# PS5a — Click-to-play: implementation (35B lane)

Every timestamp on the Results page becomes a play link. This is the
IMPLEMENTATION lane; the test suite is being written **in parallel by
another model** against the interfaces pinned below (`docs/ps/ps5b.md`).
Therefore the interface contracts here are FROZEN — implement them
exactly as written or the parallel tests will not match. Design context:
`docs/ps/2026-07-11-click-to-play-ps.md`.

## Scope fence

- **Files you may create:** `ui/video_player.py`
- **Files you may modify:** `ui/pages/Results.py`
- Touch NOTHING else. Do NOT create or edit any test file —
  `tests/test_click_to_play.py` belongs to the other lane.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN INTERFACES (the parallel test lane codes against these)

`ui/video_player.py` — streamlit-free logic except `render_player`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class VideoSource:
    path: Path                 # file to play
    kind: str                  # "original" | "strip"
    time_mapper: Callable[[float], float]  # source seconds -> playback seconds
                               # identity for "original"


def resolve_video(run_dir: Path, db) -> Optional[VideoSource]:
    """Resolution order:
    1. run_meta.video_path (db.get_meta("video_path")) if the file exists
       -> kind="original", identity mapper.
    2. run_dir/"annotated_strip.mp4" (then .avi) if it exists
       -> kind="strip", proportional mapper (see below).
    3. None.
    Never raises on a missing/moved file.
    """


def make_strip_mapper(source_dur_s: float, strip_dur_s: float) -> Callable[[float], float]:
    """Proportional mapping: t * (strip_dur_s / source_dur_s), clamped to
    [0, strip_dur_s]. source_dur_s <= 0 -> always 0.0."""


def render_player(source: VideoSource, start_s: float) -> None:
    """st.video(str(source.path), start_time=int(source.time_mapper(start_s)))
    plus st.caption stating what is playing:
    'source video @ MM:SS' or 'annotated strip (source file missing) @ MM:SS'."""
```

Strip duration for `resolve_video` comes from cv2 exactly like
`ui/pages/Results.py:1274-1278` does today (CAP_PROP_FPS fallback 4.0);
source duration from `max(frames.timestamp_ms)/1000` via the db handle.

**Session-state contract (Results.py):** a click anywhere sets
`st.session_state["player_start_s"] = <float seconds>` and
`st.session_state["player_requested"] = True`, then `st.rerun()`. One
player slot per tab renders via `render_player` when `player_requested`
is truthy. Button keys MUST be unique and deterministic:
`play_seg_{segment_id}` (transcript), `play_cap_{frame_idx}` (captions),
`play_ev_{ev_idx}` (event rows), `play_frame_{frame_idx}` (frame
inspector).

## Tasks

1. **`ui/video_player.py`** exactly per the frozen interfaces, Google
   docstrings, type hints.
2. **Transcript rows** (`_render_captions` in Results.py): replace the
   static segments dataframe with per-row layout `▶ | MM:SS–MM:SS | text`
   (`st.columns`); ▶ is `st.button(key=f"play_seg_{segment_id}")`. Keep
   the existing full-text block and metrics. Past 200 segments, paginate:
   pages of 100 behind a "Show more" button
   (`st.session_state["transcript_pages"]` counts visible pages).
3. **Captions/keyframe entries + visual-event rows**: same ▶ pattern
   wherever a start time is displayed.
4. **Frame inspector**: "▶ Play from here" button
   (key=`play_frame_{frame_idx}`) using the frame's `timestamp_ms`.
5. **Player slot**: near the top of each affected tab, render the player
   from session state via `resolve_video` + `render_player`. Missing both
   videos -> `st.info` explaining no playable video, never a crash.
6. Do not regress the Clips section or the existing plotly timeline click
   path (it keeps its embedded player; just don't break it).

## ACCEPTANCE (run, paste output — note: the new tests land in the other
lane; your gate is the existing suite + boot + wiring greps)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q          # existing suite green
grep -q "video_player" ui/pages/Results.py && echo WIRED
grep -qE "start_time" ui/video_player.py && echo SEEKS
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
/home/ashie/anaconda3/bin/python3 -m black ui/video_player.py ui/pages/Results.py
git status --porcelain                                   # only the 2 scoped files
```

## Commit message

```
Add click-to-play: every timestamp plays the video at that moment (PS5a)
```
