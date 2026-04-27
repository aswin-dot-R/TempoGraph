"""Browse and visualise past TempoGraph v2 runs from results/."""

from __future__ import annotations

import base64
import html as html_lib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT))


# ─── data access ──────────────────────────────────────────────────────────────

def _list_runs() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    runs = [p for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "tempograph.db").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_analysis(run_dir: Path) -> Optional[dict]:
    f = run_dir / "analysis.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception as e:
        st.error(f"Failed to parse analysis.json: {e}")
        return None


def _connect(run_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(run_dir / "tempograph.db"))
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def _load_run_bundle(run_dir_str: str) -> dict:
    run_dir = Path(run_dir_str)
    with _connect(run_dir) as conn:
        frames = [dict(r) for r in conn.execute(
            "SELECT frame_idx, timestamp_ms, image_path, is_keyframe, delta_score "
            "FROM frames ORDER BY frame_idx ASC"
        ).fetchall()]
        dets = [dict(r) for r in conn.execute(
            "SELECT * FROM detections"
        ).fetchall()]
        depth_rows = [dict(r) for r in conn.execute(
            "SELECT frame_idx, depth_npy_path FROM depth_frames"
        ).fetchall()]
        try:
            audio_segments = [dict(r) for r in conn.execute(
                "SELECT segment_id, start_ms, end_ms, text, no_speech_prob, avg_logprob "
                "FROM audio_segments ORDER BY start_ms ASC"
            ).fetchall()]
        except sqlite3.OperationalError:
            audio_segments = []
    det_by_frame: Dict[int, List[dict]] = {}
    for d in dets:
        det_by_frame.setdefault(d["frame_idx"], []).append(d)
    depth_by_frame: Dict[int, str] = {r["frame_idx"]: r["depth_npy_path"]
                                       for r in depth_rows}
    return {
        "frames": frames,
        "det_by_frame": det_by_frame,
        "depth_by_frame": depth_by_frame,
        "audio_segments": audio_segments,
    }


def _ts_to_seconds(ts: str) -> float:
    try:
        m, s = ts.split(":")
        return int(m) * 60 + float(s)
    except Exception:
        return 0.0


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


# ─── rendering primitives ─────────────────────────────────────────────────────

_PALETTE = [
    (66, 165, 245), (102, 187, 106), (239, 83, 80),
    (255, 167, 38), (171, 71, 188), (38, 198, 218),
    (236, 64, 122), (141, 110, 99),
]


def _color_for(key: str, seen: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if key not in seen:
        seen[key] = _PALETTE[len(seen) % len(_PALETTE)]
    return seen[key]


def _draw_detections(image_bgr: np.ndarray, dets: List[dict],
                     min_conf: float = 0.0,
                     highlight_classes: Optional[set] = None) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    seen: Dict[str, Tuple[int, int, int]] = {}
    for d in dets:
        if d["confidence"] < min_conf:
            continue
        cls = d["class_name"]
        color = _color_for(cls, seen)
        if highlight_classes and cls not in highlight_classes:
            color = (120, 120, 120)
        x1 = max(0, int(d["x1"] * w)); y1 = max(0, int(d["y1"] * h))
        x2 = min(w - 1, int(d["x2"] * w)); y2 = min(h - 1, int(d["y2"] * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        depth_str = f" d={d['mean_depth']:.2f}" if d.get("mean_depth") is not None else ""
        label = f"{cls} {d['confidence']:.2f}{depth_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _apply_depth_overlay(image_bgr: np.ndarray, depth_npy_path: Optional[str],
                         alpha: float = 0.45) -> np.ndarray:
    if not depth_npy_path:
        return image_bgr
    p = _resolve(depth_npy_path)
    if not p.exists():
        return image_bgr
    try:
        depth = np.load(str(p))
    except Exception:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)


def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


# ─── sections ─────────────────────────────────────────────────────────────────

def _render_run_picker() -> Optional[Path]:
    runs = _list_runs()
    if not runs:
        st.info(f"No runs found in {RESULTS_DIR}. Process a video first on the main page.")
        return None
    labels = [p.name for p in runs]
    label_to_path = dict(zip(labels, runs))
    choice = st.sidebar.selectbox("Run", labels, index=0)
    return label_to_path[choice]


def _render_summary(run_dir: Path, analysis: Optional[dict], bundle: dict) -> None:
    frames = bundle["frames"]
    n_dets = sum(len(v) for v in bundle["det_by_frame"].values())
    n_depth = len(bundle["depth_by_frame"])
    ts_min = min((f["timestamp_ms"] for f in frames), default=0)
    ts_max = max((f["timestamp_ms"] for f in frames), default=0)
    duration_s = (ts_max - ts_min) / 1000.0

    n_entities = len(analysis.get("entities", [])) if analysis else 0
    n_events = len(analysis.get("visual_events", [])) if analysis else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames", len(frames))
    c2.metric("Detections", n_dets)
    c3.metric("Depth maps", n_depth)
    c4.metric("Entities", n_entities)
    c5.metric("Events", n_events)
    st.caption(f"Run dir: `{run_dir}`  ·  span: {duration_s:.1f}s")

    if analysis and analysis.get("summary"):
        st.markdown("**Summary**")
        st.write(analysis["summary"])


def _render_entities_table(analysis: Optional[dict]) -> None:
    st.subheader("Entities")
    if not analysis or not analysis.get("entities"):
        st.write("_no entities_")
        return
    rows = [
        {
            "id": e.get("id", ""),
            "type": e.get("type", ""),
            "first_seen": e.get("first_seen", ""),
            "last_seen": e.get("last_seen", ""),
            "description": e.get("description", ""),
        }
        for e in analysis["entities"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _events_to_rows(events: list) -> list:
    rows = []
    for i, ev in enumerate(events):
        start = _ts_to_seconds(ev.get("start_time", "00:00"))
        end = _ts_to_seconds(ev.get("end_time", ev.get("start_time", "00:00")))
        if end <= start:
            end = start + 0.5
        ent_label = ", ".join(ev.get("entities", [])) or "—"
        ev_type = ev.get("type", "")
        if hasattr(ev_type, "value"):
            ev_type = ev_type.value
        rows.append({
            "row": ent_label,
            "type": str(ev_type),
            "start": start, "end": end,
            "description": ev.get("description", ""),
            "confidence": ev.get("confidence", 0.0),
            "label": f"{ev_type} ({ev.get('confidence', 0):.2f})",
            "ev_idx": i,
            "entity_ids": list(ev.get("entities", [])),
        })
    return rows


def _render_events_timeline(analysis: Optional[dict],
                            highlight_entity: Optional[str] = None,
                            video_duration_s: Optional[float] = None) -> None:
    events = (analysis or {}).get("visual_events", [])
    if not events:
        st.write("_no visual events_")
        return
    rows = _events_to_rows(events)
    if highlight_entity:
        rows = [r for r in rows if highlight_entity in r["entity_ids"]] or rows

    # plotly.express.timeline requires datetime; encode seconds-since-start
    # as offsets from a fixed epoch and format axis as MM:SS.
    epoch = datetime(2000, 1, 1)
    for r in rows:
        r["start_dt"] = epoch + timedelta(seconds=r["start"])
        r["end_dt"] = epoch + timedelta(seconds=r["end"])

    fig = px.timeline(
        rows,
        x_start="start_dt", x_end="end_dt", y="row", color="type",
        hover_data={
            "description": True,
            "confidence": ":.2f",
            "row": False,
            "start": ":.2f",
            "end": ":.2f",
            "start_dt": False,
            "end_dt": False,
        },
        text="label",
    )
    fig.update_yaxes(autorange="reversed", title="entity / entities")

    x_min = epoch
    x_max = epoch + timedelta(
        seconds=max(video_duration_s or 0.0,
                    max((r["end"] for r in rows), default=1.0))
    )
    fig.update_xaxes(
        title="elapsed time (mm:ss)",
        range=[x_min, x_max],
        tickformat="%M:%S",
        type="date",
    )
    fig.update_layout(
        height=max(240, 60 + 32 * len({r["row"] for r in rows})),
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title="event type",
    )
    fig.update_traces(textposition="inside", insidetextanchor="start")
    st.plotly_chart(fig, use_container_width=True)


def _render_frame_thumbnails(run_dir: Path, bundle: dict) -> None:
    frames = bundle["frames"]
    det_by_frame = bundle["det_by_frame"]
    depth_by_frame = bundle["depth_by_frame"]

    st.subheader("Frames")
    if not frames:
        st.write("_no frames_")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    show_dets = c1.checkbox("Detections", value=True, key="thumb_dets")
    show_depth = c2.checkbox("Depth heatmap", value=False, key="thumb_depth",
                             disabled=not depth_by_frame)
    keyframes_only = c3.checkbox("Keyframes only", value=False, key="thumb_kf")
    min_conf = c4.slider("Min confidence", 0.0, 1.0, 0.25, 0.05, key="thumb_conf")

    pool = [f for f in frames if (not keyframes_only or f["is_keyframe"])]
    if not pool:
        st.write("_no frames match filters_")
        return
    max_show = 60
    if len(pool) > max_show:
        st.caption(f"Showing {max_show} of {len(pool)} frames (uniform sample).")
        pool = [pool[i] for i in np.linspace(0, len(pool) - 1, max_show).astype(int)]

    cols_per_row = 4
    for i in range(0, len(pool), cols_per_row):
        row = st.columns(cols_per_row)
        for col, fr in zip(row, pool[i : i + cols_per_row]):
            img_path = _resolve(fr["image_path"])
            img = cv2.imread(str(img_path))
            if img is None:
                col.warning(f"missing: {img_path.name}")
                continue
            if show_depth:
                img = _apply_depth_overlay(img, depth_by_frame.get(fr["frame_idx"]))
            if show_dets:
                img = _draw_detections(img, det_by_frame.get(fr["frame_idx"], []),
                                       min_conf=min_conf)
            ts_s = fr["timestamp_ms"] / 1000.0
            kf = " · key" if fr["is_keyframe"] else ""
            n_d = len([d for d in det_by_frame.get(fr["frame_idx"], [])
                       if d["confidence"] >= min_conf])
            col.image(_bgr_to_png_bytes(img), width="stretch")
            col.caption(f"#{fr['frame_idx']} · t={ts_s:.2f}s · dets={n_d}{kf}")


def _render_frame_inspector(run_dir: Path, bundle: dict,
                            analysis: Optional[dict]) -> None:
    frames = bundle["frames"]
    det_by_frame = bundle["det_by_frame"]
    depth_by_frame = bundle["depth_by_frame"]

    if not frames:
        st.write("_no frames in this run_")
        return

    indices = [f["frame_idx"] for f in frames]
    chosen_idx = st.select_slider(
        "Scrub through frames",
        options=indices,
        value=indices[0],
        format_func=lambda i: f"#{i}",
    )
    fr = next(f for f in frames if f["frame_idx"] == chosen_idx)

    c1, c2, c3 = st.columns([1, 1, 2])
    show_dets = c1.checkbox("Detections", value=True, key="insp_dets")
    show_depth = c2.checkbox("Depth heatmap", value=bool(depth_by_frame),
                             key="insp_depth", disabled=not depth_by_frame)
    min_conf = c3.slider("Min confidence", 0.0, 1.0, 0.25, 0.05, key="insp_conf")

    img_path = _resolve(fr["image_path"])
    img = cv2.imread(str(img_path))
    if img is None:
        st.error(f"Cannot read frame: {img_path}")
        return
    if show_depth:
        img = _apply_depth_overlay(img, depth_by_frame.get(chosen_idx))
    if show_dets:
        img = _draw_detections(img, det_by_frame.get(chosen_idx, []),
                               min_conf=min_conf)

    ts_s = fr["timestamp_ms"] / 1000.0
    left, right = st.columns([3, 2])
    with left:
        st.image(_bgr_to_png_bytes(img), width="stretch",
                 caption=f"frame #{chosen_idx} · t={ts_s:.2f}s · "
                         f"keyframe={bool(fr['is_keyframe'])} · "
                         f"delta={fr.get('delta_score', 0):.3f}")
    with right:
        st.markdown("**Detections in this frame**")
        dets = sorted(det_by_frame.get(chosen_idx, []),
                      key=lambda d: -d["confidence"])
        if dets:
            st.dataframe(
                [{"class": d["class_name"],
                  "conf": round(d["confidence"], 3),
                  "depth": (round(d["mean_depth"], 3)
                            if d.get("mean_depth") is not None else None),
                  "bbox": f"[{d['x1']:.2f},{d['y1']:.2f},{d['x2']:.2f},{d['y2']:.2f}]"}
                 for d in dets if d["confidence"] >= min_conf],
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("_no detections_")

        st.markdown("**Active visual events at this timestamp**")
        rows = _events_to_rows((analysis or {}).get("visual_events", []))
        active = [r for r in rows if r["start"] <= ts_s <= r["end"]]
        if active:
            st.dataframe(
                [{"type": r["type"], "entities": r["row"],
                  "conf": round(r["confidence"], 2),
                  "description": r["description"]}
                 for r in active],
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("_none active_")


def _render_entity_inspector(bundle: dict, analysis: Optional[dict]) -> None:
    entities = (analysis or {}).get("entities", []) if analysis else []
    if not entities:
        st.write("_no entities to inspect_")
        return
    by_id = {e["id"]: e for e in entities}
    chosen = st.selectbox(
        "Entity",
        list(by_id.keys()),
        format_func=lambda i: f"{i} — {by_id[i].get('type', '')} — "
                              f"{by_id[i].get('description', '')[:60]}",
    )
    ent = by_id[chosen]

    c1, c2, c3 = st.columns(3)
    c1.metric("Type", ent.get("type", "—"))
    c2.metric("First seen", ent.get("first_seen", "—"))
    c3.metric("Last seen", ent.get("last_seen", "—"))
    st.write(ent.get("description", ""))

    st.markdown("#### Events involving this entity")
    rows = [r for r in _events_to_rows(analysis.get("visual_events", []))
            if chosen in r["entity_ids"]]
    if rows:
        st.dataframe(
            [{"type": r["type"], "start_s": round(r["start"], 2),
              "end_s": round(r["end"], 2), "conf": round(r["confidence"], 2),
              "co_entities": ", ".join(e for e in r["entity_ids"] if e != chosen),
              "description": r["description"]}
             for r in rows],
            use_container_width=True, hide_index=True,
        )
        st.markdown("#### Timeline (filtered)")
        frames = bundle["frames"]
        span_s = (
            (max(f["timestamp_ms"] for f in frames)
             - min(f["timestamp_ms"] for f in frames)) / 1000.0
            if frames else None
        )
        _render_events_timeline(analysis, highlight_entity=chosen,
                                video_duration_s=span_s)
    else:
        st.write("_no events reference this entity_")

    st.markdown("#### Frames during this entity's lifespan")
    fs = _ts_to_seconds(ent.get("first_seen", "00:00"))
    ls = _ts_to_seconds(ent.get("last_seen", ent.get("first_seen", "00:00")))
    frames = bundle["frames"]
    matching = [f for f in frames
                if fs <= (f["timestamp_ms"] / 1000.0) <= max(ls, fs + 0.001)]
    if not matching:
        st.write("_no frames in this lifespan window_")
        return
    show_n = min(12, len(matching))
    pick = [matching[i] for i in np.linspace(0, len(matching) - 1, show_n).astype(int)]
    cols_per_row = 4
    det_by_frame = bundle["det_by_frame"]
    for i in range(0, len(pick), cols_per_row):
        row = st.columns(cols_per_row)
        for col, fr in zip(row, pick[i : i + cols_per_row]):
            img = cv2.imread(str(_resolve(fr["image_path"])))
            if img is None:
                continue
            img = _draw_detections(img, det_by_frame.get(fr["frame_idx"], []),
                                   min_conf=0.25)
            col.image(_bgr_to_png_bytes(img), width="stretch")
            col.caption(f"#{fr['frame_idx']} · t={fr['timestamp_ms']/1000:.2f}s")


def _build_annotated_video(run_dir: Path, bundle: dict, fps: float,
                           show_dets: bool, show_depth: bool,
                           min_conf: float) -> Optional[Path]:
    frames = bundle["frames"]
    if not frames:
        return None
    first_path = _resolve(frames[0]["image_path"])
    sample = cv2.imread(str(first_path))
    if sample is None:
        return None
    h, w = sample.shape[:2]
    out_path = run_dir / "annotated_strip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        out_path = run_dir / "annotated_strip.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            return None

    det_by_frame = bundle["det_by_frame"]
    depth_by_frame = bundle["depth_by_frame"]
    progress = st.progress(0.0, text="Encoding annotated video...")
    for i, fr in enumerate(frames):
        img = cv2.imread(str(_resolve(fr["image_path"])))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        if show_depth:
            img = _apply_depth_overlay(img, depth_by_frame.get(fr["frame_idx"]))
        if show_dets:
            img = _draw_detections(img, det_by_frame.get(fr["frame_idx"], []),
                                   min_conf=min_conf)
        ts_s = fr["timestamp_ms"] / 1000.0
        cv2.putText(img, f"frame #{fr['frame_idx']}  t={ts_s:.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"frame #{fr['frame_idx']}  t={ts_s:.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
        progress.progress((i + 1) / len(frames))
    writer.release()
    progress.empty()
    return out_path


def _render_video_player(run_dir: Path, bundle: dict) -> None:
    if not bundle["frames"]:
        st.write("_no frames to encode_")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    fps = c1.slider("Playback FPS", 1.0, 12.0, 4.0, 0.5, key="vid_fps")
    show_dets = c2.checkbox("Detections", value=True, key="vid_dets")
    show_depth = c3.checkbox("Depth heatmap", value=False, key="vid_depth",
                             disabled=not bundle["depth_by_frame"])
    min_conf = c4.slider("Min confidence", 0.0, 1.0, 0.25, 0.05, key="vid_conf")

    out_path = run_dir / "annotated_strip.mp4"
    if st.button("Build / rebuild annotated video"):
        result = _build_annotated_video(run_dir, bundle, fps,
                                        show_dets, show_depth, min_conf)
        if result is None:
            st.error("Failed to encode video.")
            return
        out_path = result
    elif not out_path.exists():
        # also check the avi fallback
        if (run_dir / "annotated_strip.avi").exists():
            out_path = run_dir / "annotated_strip.avi"
        else:
            st.info("Click **Build annotated video** to render with the chosen options.")
            return

    st.caption(f"Source: `{out_path}` ({out_path.stat().st_size/1e6:.2f} MB)")
    st.video(str(out_path))


def _load_chunks(run_dir: Path) -> Optional[list]:
    f = run_dir / "chunks.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception as e:
        st.error(f"Failed to parse chunks.json: {e}")
        return None


def _render_vlm_chunks(run_dir: Path, bundle: dict) -> None:
    chunks = _load_chunks(run_dir)
    if chunks is None:
        st.info(
            "No `chunks.json` for this run. Re-run the pipeline after this update "
            "to capture per-chunk Qwen output (older runs predate the persistence)."
        )
        return
    if not chunks:
        st.write("_chunks.json is empty (VLM stage produced 0 chunks)_")
        return

    n_total = len(chunks)
    n_empty = sum(1 for c in chunks if not c.get("summary"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", n_total)
    c2.metric("Non-empty summaries", n_total - n_empty)
    c3.metric("Empty (skipped/failed)", n_empty)

    frames_by_idx = {f["frame_idx"]: f for f in bundle["frames"]}

    for ch in chunks:
        cid = ch["chunk_id"]
        frame_idxs = ch["frame_indices"]
        ts_first = frames_by_idx.get(frame_idxs[0], {}).get("timestamp_ms", 0) / 1000.0
        ts_last = frames_by_idx.get(frame_idxs[-1], {}).get("timestamp_ms", 0) / 1000.0
        empty_marker = " · ⚠ empty" if not ch.get("summary") else ""
        with st.expander(
            f"Chunk #{cid} · {len(frame_idxs)} frames · "
            f"t={ts_first:.2f}s → {ts_last:.2f}s{empty_marker}"
        ):
            st.markdown(
                f"**Summary (seed for next chunk):** "
                f"{ch.get('summary') or '_(empty)_'}"
            )

            per_frame = ch.get("per_frame_lines", {})
            if per_frame:
                st.markdown("**Per-frame lines (parsed from Qwen output):**")
                rows = []
                for fidx in frame_idxs:
                    line = per_frame.get(str(fidx), per_frame.get(fidx, ""))
                    ts_s = frames_by_idx.get(fidx, {}).get("timestamp_ms", 0) / 1000.0
                    rows.append({
                        "frame": fidx,
                        "t_s": round(ts_s, 2),
                        "qwen_caption": line or "_(no line)_",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.markdown("_no per-frame lines parsed_")

            raw = ch.get("raw_response", "")
            if raw:
                st.markdown("**Raw Qwen response (full text):**")
                st.code(raw, language="text")
            else:
                st.markdown("_no raw response captured_")


_INTERACTIVE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin: 0; padding: 0; font-family: -apple-system, system-ui, sans-serif;
         background: #0e1117; color: #e0e0e0; }
  .wrap { display: grid; grid-template-columns: minmax(0,3fr) minmax(0,2fr);
          gap: 14px; padding: 14px; }
  video { width: 100%; background: #000; border-radius: 6px; }
  #info {
    background: #1c1f24; border: 1px solid #2a2e35; border-radius: 6px;
    padding: 12px; min-height: 110px;
  }
  #info h3 { margin: 0 0 6px 0; font-size: 14px; color: #9ecbff; }
  #info p  { margin: 0; line-height: 1.4; font-size: 13px; }
  #chart { background: #1c1f24; border: 1px solid #2a2e35; border-radius: 6px;
           padding: 6px; margin: 0 14px 14px 14px; }
  .controls { padding: 0 14px 8px 14px; font-size: 12px; color: #888; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 9px;
          background: #2a2e35; color: #ddd; margin-right: 6px; font-size: 11px; }
  button { background: #2a2e35; color: #ddd; border: 1px solid #3a3f48;
           padding: 4px 10px; border-radius: 4px; cursor: pointer;
           font-size: 12px; }
  button:hover { background: #3a3f48; }
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
<div class="wrap">
  <div>
    <video id="vid" controls preload="metadata">
      <source src="data:%%MIME%%;base64,%%VIDEO_B64%%" type="%%MIME%%">
    </video>
    <div class="controls">
      <span class="pill">strip: %%STRIP_DUR%%s @ %%STRIP_FPS%% fps</span>
      <span class="pill">source span: %%SOURCE_DUR%%s</span>
      <button onclick="document.getElementById('vid').play()">▶ play</button>
      <button onclick="document.getElementById('vid').pause()">⏸ pause</button>
      <button id="loopBtn" onclick="toggleLoop()">↻ loop event: off</button>
    </div>
  </div>
  <div id="info">
    <h3>Hover or click an event on the timeline →</h3>
    <p>Hover: show one-line summary of the event. Click: jump the video to
    that segment and play. Toggle <b>loop event</b> to repeat the segment.</p>
  </div>
</div>
<div id="chart"></div>
<script>
const events = %%EVENTS_JSON%%;
const sourceDur = %%SOURCE_DUR%%;
const stripDur  = %%STRIP_DUR%%;
const vid = document.getElementById('vid');
const info = document.getElementById('info');
let loopEvent = false;
let loopRange = null;

function toggleLoop() {
  loopEvent = !loopEvent;
  document.getElementById('loopBtn').textContent =
    '↻ loop event: ' + (loopEvent ? 'on' : 'off');
}

function srcToStrip(t) {
  if (sourceDur <= 0) return 0;
  return (t / sourceDur) * stripDur;
}

function fmtTime(s) {
  const m = Math.floor(s / 60), ss = (s - m * 60).toFixed(2);
  return m.toString().padStart(2,'0') + ':' + ss.padStart(5,'0');
}

vid.addEventListener('timeupdate', () => {
  if (loopEvent && loopRange) {
    if (vid.currentTime > loopRange[1] || vid.currentTime < loopRange[0]) {
      vid.currentTime = loopRange[0];
    }
  }
});

const xStart = events.map(e => e.start);
const xEnd   = events.map(e => e.end);
const yRow   = events.map(e => e.row);
const colors = events.map(e => e.color);
const text   = events.map(e => e.label);
const customs = events.map(e => [e.description, e.start, e.end, e.confidence]);

const traces = [];
const types = [...new Set(events.map(e => e.type))];
const palette = ['#42a5f5','#66bb6a','#ef5350','#ffa726','#ab47bc',
                 '#26c6da','#ec407a','#8d6e63','#7e57c2','#26a69a'];
const typeColor = {};
types.forEach((t, i) => typeColor[t] = palette[i % palette.length]);

types.forEach(t => {
  const evs = events.filter(e => e.type === t);
  traces.push({
    type: 'bar',
    orientation: 'h',
    name: t,
    x: evs.map(e => e.end - e.start),
    base: evs.map(e => e.start),
    y: evs.map(e => e.row),
    marker: { color: typeColor[t] },
    text: evs.map(e => e.label),
    textposition: 'inside',
    insidetextanchor: 'start',
    customdata: evs.map(e => [e.description, e.start, e.end, e.confidence, e.idx]),
    hovertemplate:
      '<b>' + t + '</b><br>' +
      '%{customdata[0]}<br>' +
      '<span style="color:#9ecbff">' +
      'start %{customdata[1]:.2f}s · end %{customdata[2]:.2f}s · ' +
      'conf %{customdata[3]:.2f}</span>' +
      '<extra></extra>',
  });
});

const layout = {
  barmode: 'overlay',
  paper_bgcolor: '#1c1f24', plot_bgcolor: '#1c1f24',
  font: { color: '#e0e0e0', size: 11 },
  margin: { l: 110, r: 16, t: 8, b: 36 },
  height: Math.max(220, 40 * (new Set(yRow)).size + 80),
  xaxis: {
    title: 'video time (seconds since start)',
    range: [0, sourceDur],
    gridcolor: '#2a2e35', zerolinecolor: '#2a2e35',
  },
  yaxis: {
    autorange: 'reversed',
    gridcolor: '#2a2e35',
  },
  legend: { orientation: 'h', y: -0.25, font: { size: 11 } },
  hovermode: 'closest',
};

Plotly.newPlot('chart', traces, layout, {displaylogo: false, responsive: true});

const chart = document.getElementById('chart');

function showInfo(ev) {
  info.innerHTML =
    '<h3>' + ev.type + ' &middot; ' +
    fmtTime(ev.start) + ' &rarr; ' + fmtTime(ev.end) +
    ' &middot; conf ' + ev.confidence.toFixed(2) + '</h3>' +
    '<p>' + ev.description.replace(/</g,'&lt;') + '</p>' +
    '<p style="margin-top:6px;color:#888">entities: ' +
    (ev.entity_ids.join(', ') || '—') + '</p>';
}

chart.on('plotly_hover', d => {
  const cd = d.points[0].customdata;
  const idx = cd[4];
  showInfo(events[idx]);
});

chart.on('plotly_click', d => {
  const cd = d.points[0].customdata;
  const idx = cd[4];
  const ev = events[idx];
  showInfo(ev);
  const sStart = srcToStrip(ev.start);
  const sEnd   = srcToStrip(ev.end);
  loopRange = [sStart, Math.max(sStart + 0.3, sEnd)];
  vid.currentTime = sStart;
  vid.play();
});
</script>
</body>
</html>
"""


def _render_interactive_timeline(run_dir: Path, bundle: dict,
                                 analysis: Optional[dict]) -> None:
    events = (analysis or {}).get("visual_events", []) if analysis else []
    if not events:
        st.info("No `visual_events` in analysis.json for this run.")
        return

    frames = bundle["frames"]
    source_dur = (
        (max(f["timestamp_ms"] for f in frames)
         - min(f["timestamp_ms"] for f in frames)) / 1000.0
        if frames else 0.0
    )

    # Pick the existing annotated video, else prompt the user to build it.
    candidates = [run_dir / "annotated_strip.mp4", run_dir / "annotated_strip.avi"]
    video_path = next((p for p in candidates if p.exists()), None)
    if video_path is None:
        st.info(
            "An annotated video is required. Click below to build one with "
            "default options (4 fps, detections on, depth off)."
        )
        if st.button("Build annotated video for interactive timeline"):
            video_path = _build_annotated_video(
                run_dir, bundle, fps=4.0,
                show_dets=True, show_depth=False, min_conf=0.25,
            )
            if not video_path:
                st.error("Failed to encode video.")
                return
        else:
            return

    size_mb = video_path.stat().st_size / 1e6
    if size_mb > 30:
        st.warning(
            f"Annotated video is {size_mb:.1f} MB. Embedding into the page "
            "may be slow. Consider rebuilding at lower fps in the "
            "**Annotated video** tab."
        )

    vid_bytes = video_path.read_bytes()
    vid_b64 = base64.b64encode(vid_bytes).decode()
    mime = "video/mp4" if video_path.suffix == ".mp4" else "video/x-msvideo"

    # Probe strip duration via opencv (frame count / fps)
    cap = cv2.VideoCapture(str(video_path))
    strip_fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(bundle["frames"])
    cap.release()
    strip_dur = n_frames / strip_fps if strip_fps > 0 else 0.0

    rows = _events_to_rows(events)
    js_events = []
    for r in rows:
        js_events.append({
            "idx": r["ev_idx"],
            "type": r["type"],
            "start": float(r["start"]),
            "end": float(r["end"]),
            "row": r["row"],
            "label": r["label"],
            "description": r["description"],
            "confidence": float(r["confidence"]),
            "entity_ids": list(r["entity_ids"]),
            "color": "#42a5f5",
        })

    html_doc = (
        _INTERACTIVE_TEMPLATE
        .replace("%%MIME%%", mime)
        .replace("%%VIDEO_B64%%", vid_b64)
        .replace("%%EVENTS_JSON%%", json.dumps(js_events))
        .replace("%%SOURCE_DUR%%", f"{source_dur:.2f}")
        .replace("%%STRIP_DUR%%", f"{strip_dur:.2f}")
        .replace("%%STRIP_FPS%%", f"{strip_fps:.1f}")
    )

    # Height tied to row count: ~40 px per entity row + chrome
    n_rows = len({r["row"] for r in rows})
    height = 380 + max(220, 40 * n_rows + 80)
    st.components.v1.html(html_doc, height=height, scrolling=False)
    st.caption(
        f"Source span: {source_dur:.2f}s · Strip duration: {strip_dur:.2f}s "
        f"({strip_fps:.1f} fps) · Click an event bar to seek + play "
        "(time on the strip is mapped proportionally from source seconds)."
    )


def _render_captions(run_dir: Path, bundle: dict) -> None:
    segments = bundle.get("audio_segments", [])
    if not segments:
        st.info(
            "No audio transcript for this run. Enable **Transcribe audio** "
            "in the sidebar of the main page and re-run, or older runs "
            "predate audio support."
        )
        return

    n_seg = len(segments)
    total_chars = sum(len(s["text"]) for s in segments)
    span_s = (segments[-1]["end_ms"] - segments[0]["start_ms"]) / 1000.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Segments", n_seg)
    c2.metric("Characters", total_chars)
    c3.metric("Span", f"{span_s:.1f}s")

    st.markdown("#### Full transcript")
    full_text = " ".join(s["text"].strip() for s in segments)
    st.write(full_text)

    st.markdown("#### Segments")
    rows = []
    for s in segments:
        start_s = s["start_ms"] / 1000.0
        end_s = s["end_ms"] / 1000.0
        mm = int(start_s // 60); ss = start_s - mm * 60
        mm2 = int(end_s // 60); ss2 = end_s - mm2 * 60
        rows.append({
            "start": f"{mm:02d}:{ss:05.2f}",
            "end": f"{mm2:02d}:{ss2:05.2f}",
            "duration_s": round(end_s - start_s, 2),
            "text": s["text"].strip(),
            "no_speech_prob": (round(s["no_speech_prob"], 3)
                                if s.get("no_speech_prob") is not None else None),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    transcript_json = run_dir / "transcript.json"
    if transcript_json.exists():
        st.download_button(
            "Download transcript.json",
            data=transcript_json.read_bytes(),
            file_name="transcript.json",
            mime="application/json",
        )


def _render_artifacts(run_dir: Path) -> None:
    files = sorted(run_dir.rglob("*"))
    rows = [{"file": str(f.relative_to(run_dir)), "bytes": f.stat().st_size}
            for f in files if f.is_file()]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    graph_html = run_dir / "graph.html"
    if graph_html.exists():
        st.markdown("#### Entity graph (pyvis)")
        st.components.v1.html(graph_html.read_text(), height=600, scrolling=True)

    analysis_path = run_dir / "analysis.json"
    if analysis_path.exists():
        st.download_button(
            "Download analysis.json",
            data=analysis_path.read_bytes(),
            file_name="analysis.json",
            mime="application/json",
        )


# ─── entry ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="TempoGraph results", layout="wide")
    st.title("TempoGraph v2 — Results browser")

    run_dir = _render_run_picker()
    if run_dir is None:
        return

    analysis = _load_analysis(run_dir)
    bundle = _load_run_bundle(str(run_dir))

    (overview, frame_tab, entity_tab, vlm_tab, captions_tab,
     interactive_tab, video_tab, files_tab) = st.tabs(
        ["Overview", "Frame inspector", "Entity inspector",
         "VLM (Qwen) outputs", "Captions",
         "Interactive timeline", "Annotated video", "Files"]
    )

    frames_for_span = bundle["frames"]
    video_span_s = (
        (max(f["timestamp_ms"] for f in frames_for_span)
         - min(f["timestamp_ms"] for f in frames_for_span)) / 1000.0
        if frames_for_span else None
    )

    with overview:
        _render_summary(run_dir, analysis, bundle)
        st.divider()
        _render_entities_table(analysis)
        st.divider()
        st.subheader("Visual events timeline")
        _render_events_timeline(analysis, video_duration_s=video_span_s)
        st.divider()
        _render_frame_thumbnails(run_dir, bundle)

    with frame_tab:
        _render_frame_inspector(run_dir, bundle, analysis)

    with entity_tab:
        _render_entity_inspector(bundle, analysis)

    with vlm_tab:
        _render_vlm_chunks(run_dir, bundle)

    with captions_tab:
        _render_captions(run_dir, bundle)

    with interactive_tab:
        _render_interactive_timeline(run_dir, bundle, analysis)

    with video_tab:
        _render_video_player(run_dir, bundle)

    with files_tab:
        _render_artifacts(run_dir)


main()
