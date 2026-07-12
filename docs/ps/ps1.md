# PS1 — frame_captions schema, WAL concurrency, DB helpers

You are implementing step 1 of 4 of the Dense Temporal Captioning feature
(design: `docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`).
Work ONLY inside this scope. When done, paste every ACCEPTANCE command's
output into your final summary.

## Scope fence

- **Files you may modify:** `src/storage.py`
- **Files you may create:** `tests/test_dense_schema.py`
- Touch NOTHING else. Do not reformat unrelated code.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server, allocating GPU memory. Interpreter for all commands:
`/home/ashie/anaconda3/bin/python3`.

## Context you need (verbatim from the repo)

`src/storage.py` opens SQLite like this today:

```python
class TempoGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
```

and migrates added columns like this (keep this pattern):

```python
    def _migrate(self) -> None:
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(detections)")
        }
        if "mask_rle" not in cols:
            self._conn.execute(
                "ALTER TABLE detections ADD COLUMN mask_rle TEXT"
            )
```

## Task 1 — add the table to SCHEMA

Append to the `SCHEMA` string in `src/storage.py`:

```sql
CREATE TABLE IF NOT EXISTS frame_captions (
    frame_idx INTEGER PRIMARY KEY,
    caption TEXT NOT NULL,
    change_line TEXT,
    walker_model TEXT NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    verifier_caption TEXT,
    verifier_agrees INTEGER,
    verifier_model TEXT,
    created_at TEXT NOT NULL,
    verified_at TEXT,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_fc_escalated
    ON frame_captions(escalated, verifier_agrees);
```

`CREATE TABLE IF NOT EXISTS` inside `executescript(SCHEMA)` is the whole
migration for old DBs — they gain the table on open. Do NOT add anything to
`_migrate()` for this.

## Task 2 — WAL mode for concurrent writers

In `TempoGraphDB.__init__`, immediately after `sqlite3.connect(...)` and
before `executescript`, add:

```python
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
```

Reason (put a one-line comment above them): two threads (9B walker, 35B
verifier) write this DB at the same time through separate connections.

## Task 3 — helper methods on TempoGraphDB

Add these methods, matching the style/typing of the existing helpers
(`get_meta`, `insert_frame`, …). All timestamps ISO-8601 via
`datetime.now(timezone.utc).isoformat()` (import from stdlib `datetime`).

```python
def insert_frame_caption(
    self,
    frame_idx: int,
    caption: str,
    change_line: Optional[str],
    walker_model: str,
    escalated: bool = False,
) -> None:
    """INSERT OR REPLACE one walker row; sets created_at. Commits."""

def fetch_unverified_escalations(self, limit: int = 8) -> list:
    """Rows (sqlite3.Row) WHERE escalated=1 AND verifier_agrees IS NULL,
    ordered by frame_idx, at most `limit`."""

def save_caption_verdict(
    self,
    frame_idx: int,
    verifier_caption: str,
    verifier_agrees: bool,
    verifier_model: str,
) -> None:
    """UPDATE the row; sets verified_at. Commits."""

def count_frame_captions(self) -> Tuple[int, int, int]:
    """(total rows, escalated rows, verified rows)."""

def get_frame_caption(self, frame_idx: int):
    """One sqlite3.Row or None."""
```

Add any missing `typing` imports.

## Task 4 — tests (`tests/test_dense_schema.py`)

Use the existing test style (`tests/test_mask_persistence.py` is the model
— tmp_path DBs, plain pytest classes). Cover:

1. **Fresh DB has the table** — open `TempoGraphDB(tmp_path/"t.db")`,
   `has_table("frame_captions")` is True.
2. **Legacy DB migrates on open** — create a bare sqlite file containing
   only the old `frames` table (raw `sqlite3` calls in the test), then open
   with `TempoGraphDB`; `frame_captions` exists and old data survives.
3. **WAL is on** — `PRAGMA journal_mode` returns `"wal"`.
4. **Helper round-trip** — insert a frame row, `insert_frame_caption(...,
   escalated=True)`, `fetch_unverified_escalations()` returns it,
   `save_caption_verdict(...)`, then `fetch_unverified_escalations()` is
   empty and `count_frame_captions() == (1, 1, 1)` and
   `get_frame_caption(idx)["verifier_agrees"] == 1`.
5. **Two threads, two connections, zero locked errors** — thread A opens
   its own `TempoGraphDB` and inserts captions for frames 0..199; thread B
   opens another and calls `save_caption_verdict` on already-inserted
   escalated rows in a poll loop. Join both; assert no exception was raised
   in either thread and final counts are consistent. (Insert the 200
   `frames` rows first, from the main thread.)

## ACCEPTANCE (run these, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_dense_schema.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q     # full suite, must stay green (115 + new)
git status --porcelain                              # only the 2 scoped files
```

## Commit message

```
Add frame_captions table, WAL mode, and dense-caption DB helpers (PS1)
```
