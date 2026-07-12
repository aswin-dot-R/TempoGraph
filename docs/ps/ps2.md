# PS2 — DenseCaptionWalker: the 9B walks the frames

Step 2 of 4 of Dense Temporal Captioning
(design: `docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`).
Requires PS1 merged (`frame_captions` table + helpers exist). Paste every
ACCEPTANCE output into your final summary.

## Scope fence

- **Files you may create:** `src/modules/dense_captioner.py`,
  `tests/test_dense_walker.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server, allocating GPU memory. Tests must pass with NO server
running — mock all HTTP. Interpreter:
`/home/ashie/anaconda3/bin/python3`.

## What you are building

A class that iterates a run's `frames` table in order, calls the local
Ornith 9B vision server once per frame, and writes one `frame_captions`
row per frame via `TempoGraphDB.insert_frame_caption`. It flags big scene
changes (`escalated=1`) for the 35B verifier (built in PS3).

The 9B is an OpenAI-compatible llama-server at `http://127.0.0.1:8085`.
Payload shape (copy this; it matches `src/backends/llama_server_backend.py`):

```python
payload = {
    "model": self.model_name,
    "max_tokens": self.max_tokens,          # default 96
    "temperature": self.temperature,        # default 0.1
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
}
# POST to f"{self.base_url}/v1/chat/completions", timeout=120
# reply text = result["choices"][0]["message"]["content"]
```

## Task 1 — module skeleton

`src/modules/dense_captioner.py` with:

```python
class DenseCaptionWalker:
    def __init__(
        self,
        db_path: Path,
        base_url: str = "http://127.0.0.1:8085",
        model_name: str = "ornith-1.0-9b",
        temperature: float = 0.1,
        max_tokens: int = 96,
        escalation_percentile: float = 90.0,
        caption_similarity_floor: float = 0.3,
        request_timeout_s: float = 120.0,
        on_progress: Optional[Callable[[dict], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ): ...

    def walk(self) -> dict:
        """Caption every frame not yet in frame_captions.

        Returns {"captioned": int, "escalated": int, "skipped": int,
                 "errors": int}.
        """
```

`walk()` opens its OWN `TempoGraphDB(db_path)` (WAL handles the parallel
verifier). Google-style docstrings, full type hints, logging via
`logging.getLogger(__name__)` — log and count per-frame HTTP failures
(`errors`), never crash the walk on one bad frame.

## Task 2 — the prompt and its parser

Prompt per frame (first frame gets `prev_caption = "(first frame)"`):

```
You are watching a video one frame at a time.
Previous frame: {prev_caption}
Reply with EXACTLY two lines and nothing else:
FRAME: one sentence naming the visible objects with their colors/attributes, the surface or setting they are on, and anything happening in the background (e.g. "keys and a phone on a black table, two dogs moving in the background").
CHANGE: one short sentence on what changed since the previous frame, or "no change".
```

Parser `parse_two_lines(text) -> Tuple[str, Optional[str]]` (module-level
function, unit-testable): find the `FRAME:`-prefixed line and the
`CHANGE:`-prefixed line case-insensitively anywhere in the reply, strip
prefixes/whitespace. Fallbacks: if no `FRAME:` prefix, use the first
non-empty line as caption; if no `CHANGE:` line, change_line is None.
Empty reply → caption `"(no caption)"`.

`prev_caption` for the next iteration is the parsed caption (parsed from
the model reply, not the raw text).

## Task 3 — escalation logic

Module-level pure functions (unit-testable without HTTP):

```python
def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two lowercased captions.
    Empty∪empty → 1.0."""

def should_escalate(
    delta_score: float,
    delta_threshold: float,
    caption: str,
    prev_caption: Optional[str],
    similarity_floor: float,
) -> bool:
    """True if delta_score >= delta_threshold
    OR (prev_caption is not None and jaccard(caption, prev_caption) < similarity_floor)."""
```

`delta_threshold` is computed ONCE at walk start: the
`escalation_percentile`-th percentile of all `frames.delta_score` values in
the DB (pure-python percentile is fine; no numpy needed, but numpy is
available if you prefer). If the DB has < 10 frames, use the max
delta_score (i.e. only ties escalate) — first frame never escalates on the
similarity signal because prev is None.

## Task 4 — resume + progress

- Skip frames whose `frame_idx` already has a `frame_captions` row (count
  as `skipped`) — reruns are cheap no-ops.
- After each frame, call `on_progress({"frame_idx": ..., "done": n,
  "total": total, "escalated": m})` when set.
- Check `cancel_event.is_set()` before each frame; stop cleanly and return
  the partial counts.

## Task 5 — tests (`tests/test_dense_walker.py`)

Mock HTTP with `monkeypatch` on `requests.post` inside the module (return a
canned `choices[0].message.content` two-liner; vary it per image to drive
similarity). Fixture: tmp DB with ~20 `frames` rows pointing at tiny real
JPEGs written into tmp_path (create with cv2 or PIL — both installed).

Cover:
1. `parse_two_lines` — well-formed, missing CHANGE, prefix-less fallback,
   empty reply.
2. `jaccard` — identical → 1.0, disjoint → 0.0, empty handling.
3. `should_escalate` — delta trigger fires, similarity trigger fires,
   neither fires, first-frame (prev None) only delta can fire.
4. Percentile threshold — walker on a fixture with known delta_scores
   escalates exactly the expected frames (mock captions identical so only
   the delta signal fires).
5. Full walk — 20 frames → 20 rows, counts dict correct, prev-caption
   threading verified (assert the prompt sent for frame k contains the
   caption returned for frame k-1; capture prompts in the mock).
6. Resume — walk twice; second walk returns `captioned=0, skipped=20`.
7. Cancel — set the event after 5 frames via the progress callback; walk
   stops early and rows == frames processed.
8. HTTP error on one frame — mock raises for frame 7 only; walk completes,
   `errors == 1`, other 19 rows written.

## Optional live smoke (do NOT automate; user runs it by hand)

```bash
# with ornith-turboquant running on :8085 and a real run dir:
/home/ashie/anaconda3/bin/python3 - <<'EOF'
from pathlib import Path
from src.modules.dense_captioner import DenseCaptionWalker
w = DenseCaptionWalker(Path("results/<run>/tempograph.db"))
print(w.walk())
EOF
```

## ACCEPTANCE (run, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_dense_walker.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q      # full suite green
git status --porcelain                               # only the 2 scoped files
```

## Commit message

```
Add DenseCaptionWalker: per-frame 9B captions with change lines and escalation (PS2)
```
