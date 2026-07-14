# PS6a — Natural-language search: IMPLEMENTATION (35B lane)

"Find the part where the bird lands on the feeder." One search bar over
everything a human would look up: transcript, dense captions + change
lines + 35B opinions, detection classes, aggregated events. Results
click through to show-frame / play-span. Requires PS5 merged (uses its
session-state player contract). Tests are written in parallel
(`docs/ps/ps6b.md`) against the FROZEN CONTRACTS below.

## Scope fence

- **Files you may create:** `src/search.py`
- **Files you may modify:** `ui/pages/Results.py`
- No test files — `tests/test_search.py` belongs to the other lane.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Tests-free lane; mock nothing. Interpreter:
`/home/ashie/anaconda3/bin/python3`.

## FROZEN CONTRACTS

`src/search.py` (stdlib sqlite3 only — FTS5 ships with it):

```python
@dataclass
class SearchHit:
    timestamp_ms: int
    source_type: str          # "transcript" | "caption" | "change" | "verifier" | "detection" | "event"
    snippet: str              # matched text with the term **bolded**
    frame_idx: Optional[int]  # None for transcript/event hits
    score: float              # bm25 (lower = better, as FTS5 returns it)

def build_search_index(db_path: Path) -> int:
    """(Re)build FTS5 virtual table `search_index` inside the run DB.
    Sources: audio_segments.text; frame_captions.caption, .change_line,
    .verifier_caption (each its own row + source_type); detections.class_name
    (deduped per frame); analysis.json visual_events descriptions when the
    file exists beside the DB (timestamp from start_time MM:SS -> ms).
    Idempotent: DROP + recreate. Returns row count."""

def search(db_path: Path, query: str, limit: int = 20,
           source_filter: Optional[str] = None) -> List[SearchHit]:
    """bm25-ranked FTS5 MATCH. Empty/whitespace query -> []. Auto-builds
    the index if the table is missing. Sanitise the query for FTS5 (strip
    double quotes/operators; join terms with OR when the raw MATCH errors)."""
```

Additionally (implementation-only, NOT part of the frozen contract — the
parallel test lane does not test it):

```python
def rewrite_query(query: str, base_url: str = "http://127.0.0.1:8093",
                  timeout_s: float = 4.0) -> str:
    """Expand a natural-language query into FTS-friendly search terms via
    the tiny always-on Gemma E2B server (OpenAI-compatible llama-server).
    Prompt must start with "/no_think" and ask for ONLY a space-separated
    term list (originals + synonyms/inflections, max ~12 terms).
    ANY failure (timeout, connection, empty reply) -> return the original
    query unchanged. Never raises."""
```

The Search tab calls `search()` with the rewritten query, falling back to
the raw query when results are empty. Show the rewritten terms in a small
caption so the user sees what was searched.

UI contract: a **Search** tab in Results.py. `st.text_input` (key
`search_query`) + optional source-type selectbox; each hit renders
`MM:SS · [source_type] · snippet` with two buttons:
`find_frame_{i}` — jumps the Frame Inspector (sets the inspector's
frame-select state to the hit's frame_idx) and `find_play_{i}` — sets
`st.session_state["player_start_s"] = timestamp_ms/1000` and
`player_requested = True` (PS5's contract). "Rebuild index" button calls
`build_search_index`. Guard the whole tab: runs without the tables
render an st.info, never crash.

## ACCEPTANCE (paste output; the test file lands in the other lane)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q     # existing suite green
/home/ashie/anaconda3/bin/python3 -c "
import sqlite3; print('fts5 ok' if sqlite3.connect(':memory:').execute(
\"CREATE VIRTUAL TABLE t USING fts5(x)\") is not None else 'no fts5')"
grep -q "from src.search import\|import src.search" ui/pages/Results.py && echo WIRED
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
/home/ashie/anaconda3/bin/python3 -m black src/search.py ui/pages/Results.py
git status --porcelain   # only the 2 scoped files
```

(do not commit — gate reviewer merges both lanes)
