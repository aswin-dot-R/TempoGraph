"""Natural-language search over a TempoGraph run's SQLite store.

Uses SQLite's built-in FTS5 (ships with the stdlib) to provide BM25-ranked
search across audio transcripts, dense captions, detection classes, and
visual events from analysis.json. An optional rewrite step expands a
natural-language query into FTS-friendly terms via a local Gemma E2B
server (OpenAI-compatible llama-server).

This module is self-contained -- it only imports ``sqlite3`` from stdlib.
The ``rewrite_query`` helper makes a plain HTTP call with ``urllib``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SearchHit:
    """One row returned by :func:`search`.

    Attributes:
        timestamp_ms: Start time of the matching unit in milliseconds
            (None for event hits whose timestamp is in analysis.json only).
        source_type: One of ``"transcript" | "caption" | "change" |
            "verifier" | "detection" | "event"``.
        snippet: Matched text with the search term **bolded** via HTML
            ``<b>`` tags (produced by FTS5 ``snippet()``).
        frame_idx: The frame index the hit belongs to, or ``None`` for
            transcript/event hits that aren't frame-scoped.
        score: FTS5 BM25 score (lower = better match).
    """

    timestamp_ms: int
    source_type: str
    snippet: str
    frame_idx: Optional[int]
    score: float


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

FTS_CREATE_SQL = """
DROP TABLE IF EXISTS search_index;
CREATE VIRTUAL TABLE search_index USING fts5(
    content,
    source_type,
    timestamp_ms,
    frame_idx,
    tokenize=''
);
"""


def _parse_mmss_to_ms(time_str: str) -> int:
    """Parse an ``MM:SS`` or ``MM:SS.xx`` string into milliseconds."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            total_seconds = int(minutes) * 60 + float(seconds)
            return int(total_seconds * 1000)
    except Exception:
        pass
    return 0


def _find_analysis_json(run_dir: Path) -> Optional[dict]:
    """Locate and parse analysis.json next to the DB."""
    candidates = [
        run_dir / "analysis.json",
        run_dir / "aggregated" / "analysis.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


def build_search_index(db_path: Path) -> int:
    """(Re)build FTS5 virtual table ``search_index`` inside the run DB.

    Sources indexed:

    - ``audio_segments.text`` -> ``source_type="transcript"``
    - ``frame_captions.caption`` -> ``source_type="caption"``
    - ``frame_captions.change_line`` -> ``source_type="change"``
    - ``frame_captions.verifier_caption`` -> ``source_type="verifier"``
    - ``detections.class_name`` (deduped per frame) -> ``source_type="detection"``
    - ``analysis.json`` ``visual_events[].description`` -> ``source_type="event"``
      (timestamp derived from ``start_time`` MM:SS -> ms)

    Idempotent: DROPs + recreates the virtual table each call.

    Args:
        db_path: Path to the run's ``tempograph.db`` file.

    Returns:
        The number of rows inserted into the index.
    """
    db_path = Path(db_path)
    run_dir = db_path.parent

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(FTS_CREATE_SQL)

        n = 0

        # 1) audio_segments.text
        try:
            for row in conn.execute(
                "SELECT start_ms, text FROM audio_segments " "ORDER BY start_ms ASC"
            ):
                text = (row["text"] or "").strip()
                if text:
                    conn.execute(
                        "INSERT INTO search_index "
                        "(content, source_type, timestamp_ms, frame_idx) "
                        "VALUES (?, ?, ?, ?)",
                        (text, "transcript", row["start_ms"], None),
                    )
                    n += 1
        except sqlite3.OperationalError:
            pass

        # 2) frame_captions (caption, change_line, verifier_caption)
        try:
            fc_rows = conn.execute(
                "SELECT fc.frame_idx, fc.caption, fc.change_line, "
                "fc.verifier_caption, f.timestamp_ms "
                "FROM frame_captions fc "
                "JOIN frames f ON fc.frame_idx = f.frame_idx "
                "ORDER BY f.frame_idx ASC"
            ).fetchall()
            for row in fc_rows:
                fidx = row["frame_idx"]
                ts_ms = row["timestamp_ms"] or 0

                for col, src_type in [
                    ("caption", "caption"),
                    ("change_line", "change"),
                    ("verifier_caption", "verifier"),
                ]:
                    text = (row[col] or "").strip()
                    if text:
                        conn.execute(
                            "INSERT INTO search_index "
                            "(content, source_type, timestamp_ms, frame_idx) "
                            "VALUES (?, ?, ?, ?)",
                            (text, src_type, ts_ms, fidx),
                        )
                        n += 1
        except sqlite3.OperationalError:
            pass

        # 3) detections.class_name -- deduped per frame
        try:
            det_rows = conn.execute(
                "SELECT DISTINCT frame_idx, class_name "
                "FROM detections ORDER BY frame_idx ASC"
            ).fetchall()
            for row in det_rows:
                fidx = row["frame_idx"]
                text = (row["class_name"] or "").strip()
                if text:
                    conn.execute(
                        "INSERT INTO search_index "
                        "(content, source_type, timestamp_ms, frame_idx) "
                        "VALUES (?, ?, ?, ?)",
                        (text, "detection", None, fidx),
                    )
                    n += 1
        except sqlite3.OperationalError:
            pass

        # 4) analysis.json visual_events descriptions
        analysis = _find_analysis_json(run_dir)
        if analysis:
            try:
                events = analysis.get("visual_events", []) or []
                for ev in events:
                    desc = (ev.get("description") or "").strip()
                    if desc:
                        start_ms = _parse_mmss_to_ms(ev.get("start_time", "00:00"))
                        conn.execute(
                            "INSERT INTO search_index "
                            "(content, source_type, timestamp_ms, frame_idx) "
                            "VALUES (?, ?, ?, ?)",
                            (desc, "event", start_ms, None),
                        )
                        n += 1
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()

    return n


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _sanitize_query(query: str) -> str:
    """Strip FTS5-special characters and collapse whitespace.

    FTS5's ``MATCH`` is strict: stray quotes, backslashes, and operators
    like ``AND``, ``NOT``, ``-`` cause parse errors. We strip those and
    re-join remaining terms with ``OR`` so a multi-word query still
    matches rows containing *any* of the terms -- the behaviour the
    spec asks for as a fallback.
    """
    # Remove characters FTS5 treats as operators/punctuation.
    cleaned = re.sub(r"""['"\\[\](){}|!<>]""", " ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _try_match(conn: sqlite3.Connection, terms: str) -> str:
    """Try ``MATCH terms``; on failure fall back to OR-joined single terms.

    Returns the query string that actually succeeded (or empty string on
    total failure), so the caller can use it for the main SELECT.
    """
    # Try the cleaned query as-is first.
    try:
        conn.execute(
            "SELECT rank FROM search_index WHERE search_index MATCH ?",
            (terms,),
        ).fetchone()
        return terms
    except sqlite3.OperationalError:
        pass

    # Fallback: split into individual words and OR-join them.
    words = [w for w in terms.split() if w]
    if not words:
        return ""

    # Try the words space-joined (FTS5 treats space as AND by default).
    joined = " ".join(words)
    try:
        conn.execute(
            "SELECT rank FROM search_index WHERE search_index MATCH ?",
            (joined,),
        ).fetchone()
        return joined
    except sqlite3.OperationalError:
        pass

    # Last resort: OR-join the words.
    or_query = " OR ".join(f'"{w}"' for w in words)
    try:
        conn.execute(
            "SELECT rank FROM search_index WHERE search_index MATCH ?",
            (or_query,),
        ).fetchone()
        return or_query
    except sqlite3.OperationalError:
        return ""


def _build_snippet(conn: sqlite3.Connection, content: str) -> str:
    """Return the matched content.

    FTS5 ``snippet()`` requires MATCH context from the outer query and
    cannot be used with arbitrary ``WHERE content = ?`` filters without
    causing errors. Return the raw content as the fallback snippet.
    """
    return content


def search(
    db_path: Path,
    query: str,
    limit: int = 20,
    source_filter: Optional[str] = None,
) -> List[SearchHit]:
    """BM25-ranked FTS5 ``MATCH`` over the search index.

    Empty / whitespace-only queries return ``[]``. If the FTS5 table is
    missing, :func:`build_search_index` is triggered automatically.

    The query is sanitised to avoid FTS5 parse errors. When the raw MATCH
    fails, terms are OR-joined as a fallback.

    Args:
        db_path: Path to the run's ``tempograph.db``.
        query: Natural-language query string from the user.
        limit: Maximum number of hits to return (default 20).
        source_filter: If set, only return hits whose ``source_type``
            matches this value.

    Returns:
        A list of :class:`SearchHit`, ordered by BM25 rank (best first).
    """
    db_path = Path(db_path)
    if not query or not query.strip():
        return []

    _ensure_index(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cleaned = _sanitize_query(query)
        if not cleaned:
            return []

        match_terms = _try_match(conn, cleaned)
        if not match_terms:
            return []

        where_parts = ["search_index MATCH ?"]
        params: list = [match_terms]

        if source_filter:
            where_parts.append("source_type = ?")
            params.append(source_filter)

        params.append(int(limit))

        rows = conn.execute(
            "SELECT rank, content, source_type, timestamp_ms, frame_idx "
            "FROM search_index "
            "WHERE " + " AND ".join(where_parts) + " ORDER BY rank ASC LIMIT ?",
            params,
        ).fetchall()

        hits: List[SearchHit] = []
        for row in rows:
            hits.append(
                SearchHit(
                    timestamp_ms=row["timestamp_ms"] or 0,
                    source_type=row["source_type"],
                    snippet=_build_snippet(conn, row["content"]),
                    frame_idx=row["frame_idx"],
                    score=float(row["rank"]),
                )
            )
        return hits
    finally:
        conn.close()


def _ensure_index(db_path: Path) -> None:
    """Build the search index if it doesn't already exist."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='search_index'"
        )
        if cur.fetchone() is None:
            build_search_index(db_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query rewriting (optional, best-effort)
# ---------------------------------------------------------------------------

_REWRITE_PROMPT_TEMPLATE = (
    "/no_think You are a search-term expansion assistant. Given a natural-"
    "language query, return ONLY a space-separated list of search-friendly "
    "terms: the original words plus any synonyms, inflections, and related "
    "keywords a search engine would use. Focus on nouns and verbs. "
    "Return a maximum of 12 terms. No explanations, no quotes, no bullets -- "
    "just the terms separated by spaces.\n\n"
    "Query: {query}\n"
    "Terms:"
)


def rewrite_query(
    query: str,
    base_url: str = "http://127.0.0.1:8093",
    timeout_s: float = 4.0,
) -> str:
    """Expand a natural-language query into FTS-friendly search terms.

    Calls the always-on Gemma E2B server (OpenAI-compatible
    ``llama-server``) via ``/v1/chat/completions``. The prompt starts
    with ``/no_think`` and requests ONLY a space-separated term list
    (originals + synonyms / inflections, max ~12 terms).

    ANY failure (timeout, connection refused, empty reply, malformed JSON)
    returns the original query unchanged. Never raises.

    Args:
        query: The user's raw natural-language query.
        base_url: Base URL of the Gemma E2B llama-server.
        timeout_s: Request timeout in seconds.

    Returns:
        The rewritten (expanded) term string, or *query* unchanged on
        any failure.
    """
    if not query or not query.strip():
        return query

    prompt = _REWRITE_PROMPT_TEMPLATE.format(query=query)

    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())

        choices = body.get("choices", [])
        if not choices:
            return query
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            return query

        rewritten = re.sub(r"\s+", " ", content).strip()
        return rewritten if rewritten else query
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
        OSError,
    ):
        return query
