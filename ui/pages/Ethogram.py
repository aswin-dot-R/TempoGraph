"""Ethogram — behaviour coding with custom classes and domain context.

A separate Streamlit page that lets the user define their own behaviour
vocabulary (e.g. sit, stand, sniff, wait) and spatial/contextual hints
(e.g. "there is a feeder behind the platform"), then sends each frame
to the VLM for per-frame behaviour classification.

The results are displayed as:
  - State timeline (horizontal bars per frame)
  - Behaviour budget (pie chart)
  - Bout analysis (mean/median duration per behaviour)
  - Transition matrix (behaviour → behaviour heatmap)
  - Exportable per-frame table
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT))


# ─── data access ──────────────────────────────────────────────────

def _list_runs() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    runs = [p for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "tempograph.db").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _connect(run_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(run_dir / "tempograph.db"))
    conn.row_factory = sqlite3.Row
    return conn


def _load_frames(run_dir: Path) -> List[dict]:
    with _connect(run_dir) as conn:
        rows = conn.execute(
            "SELECT frame_idx, timestamp_ms, image_path, is_keyframe, delta_score "
            "FROM frames ORDER BY frame_idx ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _load_detections(run_dir: Path) -> Dict[int, List[dict]]:
    with _connect(run_dir) as conn:
        rows = conn.execute("SELECT * FROM detections").fetchall()
    by_frame: Dict[int, List[dict]] = {}
    for d in [dict(r) for r in rows]:
        by_frame.setdefault(d["frame_idx"], []).append(d)
    return by_frame


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


# ─── VLM ethogram prompt ─────────────────────────────────────────

ETHOGRAM_PROMPT_TEMPLATE = """/no_think
You are a behavioural scientist coding video observations of a subject.

## Subject
{subject_description}

## Valid behaviour classes (use ONLY these)
{behaviour_classes}

## Spatial / contextual notes from the researcher
{context_notes}

## Previous labels (for context continuity)
{previous_labels}

## Frames to classify
{frame_block}

## Task
For EACH frame listed above, classify the PRIMARY behaviour of the subject.
Pick exactly ONE behaviour from the valid classes.
Rate your confidence 0.0–1.0.

Output ONLY a JSON array, one object per frame (no markdown, no explanation):
[{{"frame": <idx>, "behavior": "<class>", "confidence": 0.85, "note": "brief reason"}}, ...]
"""

ETHOGRAM_PROMPT_SINGLE = """/no_think
You are a behavioural scientist coding video observations of a subject.

## Subject
{subject_description}

## Valid behaviour classes (use ONLY these)
{behaviour_classes}

## Spatial / contextual notes from the researcher
{context_notes}

## Previous labels (for context continuity)
{previous_labels}

## YOLO detections in this frame
{detections}

## Task
Look at this frame and classify the PRIMARY behaviour of the subject.
Pick exactly ONE behaviour from the valid classes above.
Rate your confidence 0.0–1.0.
Add a brief note explaining why you chose this label.

Output ONLY this JSON (no markdown, no explanation):
{{"behavior": "<class>", "confidence": 0.85, "note": "brief reason"}}
"""


def _format_detections(dets: List[dict]) -> str:
    if not dets:
        return "(no detections)"
    parts = []
    for d in dets:
        base = (
            f"{d['class_name']} at [{d['x1']:.2f},{d['y1']:.2f},"
            f"{d['x2']:.2f},{d['y2']:.2f}] conf={d['confidence']:.2f}"
        )
        parts.append(base)
    return "; ".join(parts)


def _encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vlm(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str,
) -> dict:
    """Send a single frame to the VLM and parse the JSON response."""
    import requests

    payload = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    content = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    # Parse JSON from response
    try:
        # Try direct parse first
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown/text
        import re
        m = re.search(r"\{[^}]+\}", content)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"behavior": "unknown", "confidence": 0.0, "note": f"unparseable: {content[:100]}"}


def _format_previous_labels(labels: List[dict]) -> str:
    """Format recent ethogram labels as context for the next VLM call."""
    if not labels:
        return "(this is the start — no previous labels)"
    lines = []
    for lab in labels[-10:]:
        lines.append(
            f"  frame {lab['frame_idx']}: {lab['behavior']} "
            f"(conf={lab.get('confidence', 0):.2f})"
        )
    return "\n".join(lines)


def _run_ethogram_batch(
    batch_indices: List[int],
    frame_map: Dict[int, dict],
    det_by_frame: Dict[int, List[dict]],
    behavior_classes: List[str],
    subject_desc: str,
    context_notes: str,
    recent_labels: List[dict],
    vlm_url: str,
    vlm_model: str,
) -> dict:
    """Run ethogram classification on a batch of frames.

    Sends multiple frames in one VLM call for temporal context.
    Falls back to single-frame calls if batch parsing fails.

    Returns:
        {"labels": [...], "prompt_tokens": N, "total_tokens": N}
    """
    import requests as _req

    prev_labels_text = _format_previous_labels(recent_labels)
    lower_map = {c.lower(): c for c in behavior_classes}

    # Try batched multi-frame call
    if len(batch_indices) > 1:
        images_b64 = []
        frame_lines = []
        valid_indices = []
        for fidx in batch_indices:
            fr = frame_map.get(fidx)
            if not fr:
                continue
            img_path = _resolve(fr["image_path"])
            if not img_path.exists():
                continue
            images_b64.append(_encode_image_b64(img_path))
            dets = det_by_frame.get(fidx, [])
            ts_s = fr["timestamp_ms"] / 1000.0
            frame_lines.append(
                f"[frame {fidx} — t={ts_s:.2f}s] YOLO: {_format_detections(dets)}"
            )
            valid_indices.append(fidx)

        if valid_indices:
            prompt = ETHOGRAM_PROMPT_TEMPLATE.format(
                subject_description=subject_desc,
                behaviour_classes="\n".join(f"  - {c}" for c in behavior_classes),
                context_notes=context_notes,
                previous_labels=prev_labels_text,
                frame_block="\n".join(frame_lines),
            )

            try:
                # Build multi-image payload
                content_items: List[dict] = [{"type": "text", "text": prompt}]
                for b64 in images_b64:
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

                payload = {
                    "model": vlm_model,
                    "max_tokens": 200 * len(valid_indices),
                    "temperature": 0.1,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": content_items}],
                }

                response = _req.post(
                    f"{vlm_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                resp_json = response.json()

                raw_content = (
                    resp_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                usage = resp_json.get("usage", {})

                # Parse JSON array from response
                import re
                parsed = None
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError:
                    m = re.search(r"\[.*\]", raw_content, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except json.JSONDecodeError:
                            pass

                if isinstance(parsed, list) and len(parsed) > 0:
                    labels = []
                    for item in parsed:
                        fidx = int(item.get("frame", valid_indices[len(labels)]
                                            if len(labels) < len(valid_indices)
                                            else valid_indices[-1]))
                        behavior = str(item.get("behavior", "unknown"))
                        if behavior not in behavior_classes:
                            behavior = lower_map.get(behavior.lower(),
                                                     behavior_classes[-1])
                        labels.append({
                            "frame_idx": fidx,
                            "behavior": behavior,
                            "confidence": float(item.get("confidence", 0)),
                            "note": str(item.get("note", "")),
                        })
                    return {
                        "labels": labels,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }

            except Exception:
                pass  # Fall through to single-frame mode

    # Single-frame fallback (or batch_size == 1)
    labels = []
    total_prompt = 0
    total_tokens = 0

    for fidx in batch_indices:
        fr = frame_map.get(fidx)
        if not fr:
            continue
        img_path = _resolve(fr["image_path"])
        if not img_path.exists():
            continue

        try:
            dets = det_by_frame.get(fidx, [])
            prompt = ETHOGRAM_PROMPT_SINGLE.format(
                subject_description=subject_desc,
                behaviour_classes="\n".join(f"  - {c}" for c in behavior_classes),
                context_notes=context_notes,
                previous_labels=prev_labels_text,
                detections=_format_detections(dets),
            )

            result = _call_vlm(vlm_url, vlm_model, prompt, _encode_image_b64(img_path))
            behavior = str(result.get("behavior", "unknown"))
            if behavior not in behavior_classes:
                behavior = lower_map.get(behavior.lower(), behavior_classes[-1])

            labels.append({
                "frame_idx": fidx,
                "behavior": behavior,
                "confidence": float(result.get("confidence", 0)),
                "note": str(result.get("note", "")),
            })
        except Exception as e:
            labels.append({
                "frame_idx": fidx,
                "behavior": "error",
                "confidence": 0.0,
                "note": str(e)[:100],
            })

    return {"labels": labels, "prompt_tokens": total_prompt, "total_tokens": total_tokens}


# ─── visualisation ────────────────────────────────────────────────

_BEHAVIOR_COLORS = [
    "#42a5f5", "#66bb6a", "#ef5350", "#ffa726", "#ab47bc",
    "#26c6da", "#ec407a", "#8d6e63", "#7e57c2", "#26a69a",
    "#d4e157", "#78909c", "#ff7043", "#5c6bc0", "#29b6f6",
]


def _assign_colors(classes: List[str]) -> Dict[str, str]:
    return {c: _BEHAVIOR_COLORS[i % len(_BEHAVIOR_COLORS)]
            for i, c in enumerate(classes)}


def _render_state_timeline(
    labels: List[dict],
    frames: List[dict],
    color_map: Dict[str, str],
) -> None:
    """Horizontal bar timeline showing behaviour state per frame."""
    if not labels:
        st.write("_no labels to display_")
        return

    frame_ts = {f["frame_idx"]: f["timestamp_ms"] / 1000.0 for f in frames}

    # Build Gantt-like data: merge consecutive same-behaviour frames into bouts
    bouts = []
    current_behavior = None
    bout_start = 0.0
    bout_start_idx = 0

    sorted_labels = sorted(labels, key=lambda l: l["frame_idx"])
    for i, lab in enumerate(sorted_labels):
        ts = frame_ts.get(lab["frame_idx"], 0.0)
        if lab["behavior"] != current_behavior:
            if current_behavior is not None:
                bouts.append({
                    "behavior": current_behavior,
                    "start": bout_start,
                    "end": ts,
                    "frames": i - bout_start_idx,
                    "frame_indices": [l["frame_idx"] for l in sorted_labels[bout_start_idx:i]],
                })
            current_behavior = lab["behavior"]
            bout_start = ts
            bout_start_idx = i

    # Close last bout
    if current_behavior is not None:
        last_ts = frame_ts.get(sorted_labels[-1]["frame_idx"], bout_start + 0.5)
        bouts.append({
            "behavior": current_behavior,
            "start": bout_start,
            "end": max(last_ts, bout_start + 0.1),
            "frames": len(sorted_labels) - bout_start_idx,
            "frame_indices": [l["frame_idx"] for l in sorted_labels[bout_start_idx:]],
        })

    if not bouts:
        return

    fig = go.Figure()
    for behavior in color_map:
        b_bouts = [b for b in bouts if b["behavior"] == behavior]
        if not b_bouts:
            continue
        fig.add_trace(go.Bar(
            name=behavior,
            x=[b["end"] - b["start"] for b in b_bouts],
            base=[b["start"] for b in b_bouts],
            y=["subject"] * len(b_bouts),
            orientation="h",
            marker_color=color_map[behavior],
            text=[f"{behavior} ({b['frames']}f)" for b in b_bouts],
            textposition="inside",
            customdata=[",".join(map(str, b["frame_indices"])) for b in b_bouts],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "start: %{base:.2f}s<br>"
                "duration: %{x:.2f}s<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Behaviour state timeline",
        xaxis_title="time (s)",
        legend=dict(orientation="h", y=-0.3),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c1f24",
        font=dict(color="#e0e0e0"),
    )
    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    if selection and selection.get("selection", {}).get("points"):
        pt = selection["selection"]["points"][0]
        if "customdata" in pt and pt["customdata"]:
            cdata = pt["customdata"]
            if isinstance(cdata, list):
                cdata = cdata[0]
            if cdata:
                indices = [int(x) for x in str(cdata).split(",") if x]
                _display_selected_frames(frames, labels, indices, title=f"Frames for {pt.get('text', 'Bout')}")

def _display_selected_frames(frames: List[dict], labels: List[dict], indices: List[int], title: str = "Selected Frames") -> None:
    if not indices:
        return
    st.markdown(f"**{title}**")
    valid = [f for f in frames if f["frame_idx"] in indices]
    if not valid:
        st.write("No images found.")
        return
    
    behavior_map = {l["frame_idx"]: l["behavior"] for l in labels}
    
    cols = st.columns(min(len(valid), 4))
    for i, f in enumerate(valid[:20]):
        with cols[i % len(cols)]:
            st.image(f["image_path"], use_container_width=True)
            beh = behavior_map.get(f["frame_idx"], "")
            st.caption(f"Frame {f['frame_idx']} ({f['timestamp_ms']/1000:.2f}s)<br>**{beh}**", unsafe_allow_html=True)
    if len(valid) > 20:
        st.caption(f"...and {len(valid)-20} more frames")


def _render_behavior_budget(labels: List[dict], color_map: Dict[str, str]) -> None:
    """Pie chart of time spent in each behaviour."""
    counts = Counter(l["behavior"] for l in labels)
    if not counts:
        return

    behaviors = list(counts.keys())
    values = list(counts.values())
    colors = [color_map.get(b, "#888") for b in behaviors]

    fig = px.pie(
        names=behaviors,
        values=values,
        title="Behaviour budget",
        color_discrete_sequence=colors,
    )
    fig.update_layout(
        height=350,
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_bout_analysis(
    labels: List[dict],
    frames: List[dict],
) -> None:
    """Table showing mean/median/count per behaviour."""
    if not labels:
        return

    frame_ts = {f["frame_idx"]: f["timestamp_ms"] / 1000.0 for f in frames}
    sorted_labels = sorted(labels, key=lambda l: l["frame_idx"])

    # Compute bouts
    bouts: Dict[str, List[float]] = defaultdict(list)
    current = None
    bout_start = 0.0
    for lab in sorted_labels:
        ts = frame_ts.get(lab["frame_idx"], 0.0)
        if lab["behavior"] != current:
            if current is not None:
                bouts[current].append(ts - bout_start)
            current = lab["behavior"]
            bout_start = ts
    if current is not None:
        last_ts = frame_ts.get(sorted_labels[-1]["frame_idx"], bout_start)
        bouts[current].append(max(last_ts - bout_start, 0.1))

    rows = []
    for behavior, durations in sorted(bouts.items()):
        rows.append({
            "Behaviour": behavior,
            "Bouts": len(durations),
            "Total (s)": round(sum(durations), 2),
            "Mean (s)": round(np.mean(durations), 2),
            "Median (s)": round(np.median(durations), 2),
            "Min (s)": round(min(durations), 2),
            "Max (s)": round(max(durations), 2),
        })

    st.subheader("Bout analysis")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_transition_matrix(labels: List[dict], frames: List[dict]) -> None:
    """Heatmap of behaviour→behaviour transitions."""
    if len(labels) < 2:
        return

    sorted_labels = sorted(labels, key=lambda l: l["frame_idx"])
    behaviors = sorted(set(l["behavior"] for l in sorted_labels))
    n = len(behaviors)
    idx = {b: i for i, b in enumerate(behaviors)}
    matrix = np.zeros((n, n), dtype=int)
    customdata = [["" for _ in range(n)] for _ in range(n)]

    for i in range(len(sorted_labels) - 1):
        a = sorted_labels[i]["behavior"]
        b = sorted_labels[i + 1]["behavior"]
        if a != b:  # only count actual transitions
            idx_a = idx[a]
            idx_b = idx[b]
            matrix[idx_a][idx_b] += 1
            trans_idx = str(sorted_labels[i+1]["frame_idx"])
            if customdata[idx_a][idx_b]:
                customdata[idx_a][idx_b] += "," + trans_idx
            else:
                customdata[idx_a][idx_b] = trans_idx

    fig = px.imshow(
        matrix,
        labels=dict(x="To", y="From", color="Count"),
        x=behaviors,
        y=behaviors,
        title="Behaviour transition matrix",
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        height=max(300, 50 * n + 100),
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    fig.update_traces(customdata=customdata)
    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    if selection and selection.get("selection", {}).get("points"):
        pt = selection["selection"]["points"][0]
        if "customdata" in pt and pt["customdata"]:
            cdata = pt["customdata"]
            if isinstance(cdata, list):
                cdata = cdata[0]
            if cdata:
                indices = [int(x) for x in str(cdata).split(",") if x]
                # The transition happens at the given frame index, let's show that frame + the one before it if we want, but just showing the transition frames is fine
                from_b = pt.get("y", "Unknown")
                to_b = pt.get("x", "Unknown")
                _display_selected_frames(frames, labels, indices, title=f"Transitions: {from_b} → {to_b}")


# ─── main page ────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="TempoGraph Ethogram", layout="wide")
    st.title("🐾 Ethogram — Behaviour Coding")
    st.caption(
        "Define your behaviour vocabulary and spatial context, then let the VLM "
        "classify each frame. Results are stored per-profile and can be exported."
    )

    # ── Step 1: select video run ──
    st.subheader("Step 1 — Select or upload video")

    # Upload new video section
    import tempfile
    import os
    import subprocess as _sp
    from src.models import PipelineConfig
    from src.pipeline_v2 import PipelineV2

    input_mode = st.radio(
        "Video source",
        ["Local file path (recommended for large files)", "Upload (< 2 GB)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    novel_video_path = None
    novel_video_name = None

    if input_mode.startswith("Local"):
        local_path = st.text_input(
            "Path to video file",
            placeholder="/home/ashie/videos/camera0_4k.mkv",
            help="Absolute path on your filesystem. "
                 "No RAM overhead — the file is read directly by OpenCV/FFmpeg.",
        )
        if local_path and os.path.isfile(local_path):
            novel_video_name = os.path.basename(local_path)
            ext = os.path.splitext(local_path)[1].lower()
            if ext != ".mp4":
                fixed = os.path.join(
                    tempfile.gettempdir(),
                    os.path.splitext(novel_video_name)[0] + "_fixed.mp4",
                )
                if not os.path.exists(fixed):
                    with st.spinner(f"Remuxing {novel_video_name} → MP4..."):
                        _sp.run(
                            ["ffmpeg", "-y", "-i", local_path,
                             "-c", "copy", "-fflags", "+genpts", fixed],
                            capture_output=True,
                        )
                novel_video_path = fixed
            else:
                novel_video_path = local_path
        elif local_path:
            st.error(f"File not found: `{local_path}`")
    else:
        uploaded = st.file_uploader(
            "Upload a novel video to code (< 2 GB)",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded is not None:
            novel_video_name = uploaded.name
            ext = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(uploaded.read())
                raw_path = f.name
            fixed = raw_path + "_fixed.mp4"
            _sp.run(
                ["ffmpeg", "-y", "-i", raw_path,
                 "-c", "copy", "-fflags", "+genpts", fixed],
                capture_output=True,
            )
            novel_video_path = fixed

    if novel_video_path is not None:
        run_name = os.path.splitext(novel_video_name)[0]
        run_out = RESULTS_DIR / run_name
        # Only preprocess if not already done
        if not (run_out / "tempograph.db").exists():
            # Preprocessing controls
            with st.expander("⚙ Preprocessing settings", expanded=True):
                pp_col1, pp_col2, pp_col3 = st.columns(3)
                with pp_col1:
                    pp_camera = st.selectbox(
                        "Camera mode",
                        ["static", "moving", "auto"],
                        index=0,
                        help="static: fixed camera (pixel delta). "
                             "moving: handheld/drone (motion-compensated). "
                             "auto: detect from first 30 frames.",
                    )
                    pp_fps = st.slider(
                        "Sample FPS", 0.5, 10.0, 1.0, 0.5,
                        help="How many frames per second to sample for YOLO. "
                             "Higher = more frames but slower processing.",
                    )
                with pp_col2:
                    pp_threshold = st.slider(
                        "Keyframe threshold (× σ)", 0.5, 3.0, 1.0, 0.1,
                        help="Controls how sensitive keyframe detection is. "
                             "Lower = more keyframes. Higher = only big changes.",
                    )
                    pp_confidence = st.slider(
                        "YOLO confidence", 0.1, 0.9, 0.5, 0.05,
                        help="Minimum detection confidence for YOLO bounding boxes.",
                    )
                with pp_col3:
                    pp_yolo_size = st.selectbox(
                        "YOLO model",
                        ["n", "s", "m", "l"],
                        index=0,
                        format_func=lambda s: {
                            "n": "Nano (fastest)",
                            "s": "Small (balanced)",
                            "m": "Medium (accurate)",
                            "l": "Large (best)",
                        }[s],
                        help="Larger models are more accurate but slower.",
                    )

            if st.button("⚙ Preprocess this video (extract frames + YOLO)"):
                with st.status("Preprocessing novel video (frames + YOLO)...", expanded=True) as status:
                    stage_log = st.empty()
                    stage_state: dict = {"lines": []}
                    def _icon(event: str) -> str:
                        return {"start": "▶", "done": "✓", "error": "✗", "skipped": "⏭"}.get(event, "·")

                    def _on_stage(name: str, event: str, info: dict) -> None:
                        suffix = ""
                        if event == "done":
                            if "elapsed_s" in info:
                                suffix += f" ({info['elapsed_s']}s)"
                        if event == "progress":
                            # In-place progress update
                            if name == "Frame selection":
                                detail = (
                                    f"  ↳ Scanning: frame {info.get('frame', '?')}/{info.get('total', '?')}"
                                    f" @ {info.get('fps', 0)} fps — ETA {info.get('eta_s', 0)}s"
                                )
                            elif name == "YOLO detection":
                                detail = (
                                    f"  ↳ YOLO: {info.get('step', 0)}/{info.get('total', 0)} frames"
                                    f" — {info.get('detections', 0)} dets"
                                    f" @ {info.get('fps', 0)} fps — ETA {info.get('eta_s', 0)}s"
                                )
                            else:
                                detail = f"  ↳ {name}: {info}"
                            if stage_state["lines"] and stage_state["lines"][-1].startswith("  ↳"):
                                stage_state["lines"][-1] = detail
                            else:
                                stage_state["lines"].append(detail)
                            stage_log.code("\n".join(stage_state["lines"]), language="text")
                            return
                        if event in ("start", "done", "skipped", "error"):
                            if stage_state["lines"] and stage_state["lines"][-1].startswith("  ↳"):
                                stage_state["lines"].pop()
                            stage_state["lines"].append(f"{_icon(event)} {name} — {event}{suffix}")
                        status.update(label=f"{name} ({event})")
                        stage_log.code("\n".join(stage_state["lines"]), language="text")

                    from src.models import CameraMode
                    camera_map = {"static": CameraMode.STATIC, "moving": CameraMode.MOVING, "auto": CameraMode.AUTO}

                    config = PipelineConfig(
                        video_path=novel_video_path,
                        output_dir=str(run_out),
                        max_frames=10000,
                        confidence=pp_confidence,
                    )
                    pipe = PipelineV2(
                        config,
                        skip_vlm=True,
                        audio_enabled=False,
                        depth_enabled=False,
                        camera_mode=camera_map[pp_camera],
                        yolo_fps=pp_fps,
                        threshold_mult=pp_threshold,
                        yolo_size=pp_yolo_size,
                        on_stage=_on_stage,
                    )
                    try:
                        pipe.run()
                        status.update(label="Preprocessing complete!", state="complete")
                        st.session_state.ethogram_run = run_out
                        st.rerun()
                    except Exception as e:
                        status.update(label=f"Error: {e}", state="error")
        else:
            st.success(f"✓ `{novel_video_name}` already preprocessed — selecting it.")
            st.session_state.ethogram_run = run_out

    runs = _list_runs()
    if not runs:
        st.info(f"No runs found in `{RESULTS_DIR}`. Please provide a video above and preprocess it.")
        return

    # Show all runs as cards
    run_cols = st.columns(min(len(runs), 4))
    if "ethogram_run" not in st.session_state:
        st.session_state.ethogram_run = None

    for i, run_path in enumerate(runs):
        col = run_cols[i % len(run_cols)]
        name = run_path.name
        is_selected = (
            st.session_state.ethogram_run is not None
            and st.session_state.ethogram_run.name == name
        )

        # Quick stats
        try:
            mtime = run_path.stat().st_mtime
            from datetime import datetime
            time_str = datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M")
        except Exception:
            time_str = ""

        n_fr = 0
        try:
            import sqlite3 as _sql
            with _sql.connect(str(run_path / "tempograph.db")) as _c:
                _c.row_factory = _sql.Row
                n_fr = _c.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        except Exception:
            pass

        bg = "#1a3a5c" if is_selected else "#1c1f24"
        border = "2px solid #42a5f5" if is_selected else "1px solid #2a2e35"
        check = " ✓" if is_selected else ""

        with col:
            st.markdown(
                f'<div style="background:{bg};border:{border};border-radius:10px;'
                f'padding:12px;text-align:center;margin-bottom:6px">'
                f'<div style="font-size:14px;font-weight:700;color:#e0e0e0">'
                f'🎬 {name}{check}</div>'
                f'<div style="font-size:11px;color:#888;margin-top:4px">'
                f'{n_fr} frames · {time_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Select" if not is_selected else "Selected ✓",
                key=f"ethogram_sel_{name}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.ethogram_run = run_path
                st.rerun()

    if st.session_state.ethogram_run is None:
        st.warning("👆 Select a processed video above to begin ethogram coding.")
        return

    run_dir = st.session_state.ethogram_run

    # Show a preview thumbnail of the selected run
    frames = _load_frames(run_dir)
    if not frames:
        st.warning("No frames in this run.")
        return

    # Show first frame as preview
    first_frame = frames[0]
    preview_path = _resolve(first_frame["image_path"])
    if preview_path.exists():
        st.image(
            str(preview_path), width=320,
            caption=f"Selected: {run_dir.name} — {len(frames)} frames",
        )

    st.divider()
    st.subheader("Step 2 — Configure behaviour classes")

    det_by_frame = _load_detections(run_dir)

    # ── user configuration ──
    st.sidebar.header("Ethogram configuration")

    subject_desc = st.sidebar.text_input(
        "Subject description",
        value="a dog on a testing platform",
        help="What/who are you observing? E.g. 'a beagle on a scent platform'",
    )

    behavior_input = st.sidebar.text_area(
        "Behaviour classes (one per line)",
        value="sit\nstand\nwalk\nsniff\nwait\nindication\nother",
        height=200,
        help="List every valid behaviour class the model should choose from. "
             "One per line. The model will pick exactly one per frame.",
    )
    behavior_classes = [
        b.strip() for b in behavior_input.strip().split("\n")
        if b.strip()
    ]

    context_notes = st.sidebar.text_area(
        "Spatial / contextual notes",
        value=(
            "There is a scent feeder mounted at the back of the platform. "
            "When the dog turns toward the feeder and holds position, "
            "that is an 'indication' behaviour. "
            "The dog may sniff the ground or air before indicating."
        ),
        height=200,
        help="Tell the model about the setup — landmarks, equipment, "
             "what specific behaviours look like. Be specific!",
    )

    profile_name = st.sidebar.text_input(
        "Profile name",
        value="default",
        help="Name this configuration. Different profiles store separate results.",
    )

    st.sidebar.divider()

    # Frame sampling
    all_frame_indices = [f["frame_idx"] for f in frames]
    frame_mode = st.sidebar.radio(
        "Frames to analyse",
        ["Keyframes only", "Every Nth frame", "All frames"],
        index=0,
    )
    if frame_mode == "Keyframes only":
        target_indices = [f["frame_idx"] for f in frames if f["is_keyframe"]]
        if not target_indices:
            target_indices = all_frame_indices[:20]
    elif frame_mode == "Every Nth frame":
        n = st.sidebar.slider("N (every Nth)", 2, 20, 5)
        target_indices = all_frame_indices[::n]
    else:
        target_indices = all_frame_indices

    st.sidebar.metric("Frames to code", len(target_indices))

    # VLM settings
    st.sidebar.divider()
    st.sidebar.subheader("VLM settings")
    vlm_url = st.sidebar.text_input("VLM URL", "http://127.0.0.1:8082")
    vlm_model = st.sidebar.text_input("Model", "Qwen3.5-9B-Q8_0.gguf")

    # ── existing results ──
    with _connect(run_dir) as conn:
        # Ensure ethogram tables exist (for older DBs)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ethogram_labels (
                label_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_idx INTEGER NOT NULL,
                behavior TEXT NOT NULL,
                confidence REAL,
                note TEXT,
                profile_name TEXT NOT NULL DEFAULT 'default'
            );
            CREATE INDEX IF NOT EXISTS idx_ethogram_frame ON ethogram_labels(frame_idx);
            CREATE INDEX IF NOT EXISTS idx_ethogram_profile ON ethogram_labels(profile_name);
        """)
        existing = [
            dict(r) for r in conn.execute(
                "SELECT * FROM ethogram_labels WHERE profile_name = ? "
                "ORDER BY frame_idx ASC",
                (profile_name,),
            ).fetchall()
        ]
        profiles = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT profile_name FROM ethogram_labels"
            ).fetchall()
        ]

    if profiles:
        st.sidebar.caption(f"Existing profiles: {', '.join(profiles)}")

    # ── main panel ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Behaviour classes", len(behavior_classes))
    col2.metric("Frames to code", len(target_indices))
    col3.metric("Existing labels", len(existing))

    # Show the defined classes
    color_map = _assign_colors(behavior_classes)
    class_html = " ".join(
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{color_map[c]};color:#fff;margin:2px;font-size:13px">{c}</span>'
        for c in behavior_classes
    )
    st.markdown(f"**Behaviour classes:** {class_html}", unsafe_allow_html=True)

    if context_notes.strip():
        st.info(f"**Context:** {context_notes.strip()}")

    # ── run analysis ──
    run_col, clear_col = st.columns([2, 1])

    if run_col.button("▶ Run ethogram analysis", type="primary"):
        if not behavior_classes:
            st.error("Define at least one behaviour class.")
            return

        # Clear existing labels for this profile
        with _connect(run_dir) as conn:
            conn.execute(
                "DELETE FROM ethogram_labels WHERE profile_name = ?",
                (profile_name,),
            )
            conn.commit()

        frame_map = {f["frame_idx"]: f for f in frames}
        progress = st.progress(0.0, text="Starting ethogram analysis...")
        results: List[dict] = []

        # ── Check/start VLM service ──
        import requests as _req
        try:
            _req.get(f"{vlm_url}/v1/models", timeout=2)
        except Exception:
            progress.progress(0.0, text="Starting VLM service (qwen35-turboquant.service)...")
            import subprocess
            try:
                subprocess.run(
                    ["systemctl", "--user", "start", "qwen35-turboquant.service"],
                    check=True, capture_output=True, timeout=15
                )
                # Wait for it to become ready
                ready = False
                for _ in range(60):
                    import time
                    time.sleep(1.0)
                    try:
                        if _req.get(f"{vlm_url}/v1/models", timeout=1).status_code == 200:
                            ready = True
                            break
                    except Exception:
                        pass
                if not ready:
                    st.error(f"VLM service did not become reachable at {vlm_url}")
                    return
            except Exception as e:
                st.warning(f"Failed to autostart VLM service: {e}")

        # ── context-aware batching ──
        # Query n_ctx from the VLM server for dynamic batch sizing
        n_ctx = 100096
        try:
            props = _req.get(f"{vlm_url}/props", timeout=5).json()
            n_ctx = int(
                props.get("default_generation_settings", {}).get("n_ctx", 0)
            ) or 100096
        except Exception:
            pass

        budget = int(n_ctx * 0.80)
        est_tokens_per_image = 1500
        prompt_overhead = 800
        completion_per_frame = 80
        cumulative_tokens = 0

        # Recent labels for context carry-over
        recent_labels: List[dict] = []  # last N labels for context
        MAX_CONTEXT_LABELS = 10

        idx = 0
        while idx < len(target_indices):
            # Dynamic batch size based on remaining budget
            prev_labels_text = _format_previous_labels(recent_labels[-MAX_CONTEXT_LABELS:])
            prev_labels_tokens = max(50, len(prev_labels_text) // 3)
            available = budget - prompt_overhead - prev_labels_tokens
            batch_size = max(
                1, min(8, available // max(1, est_tokens_per_image + completion_per_frame))
            )

            batch_indices = target_indices[idx: idx + batch_size]
            progress.progress(
                (idx + 1) / len(target_indices),
                text=f"Batch starting at frame {batch_indices[0]} "
                     f"({idx+1}/{len(target_indices)}, batch={len(batch_indices)})",
            )

            batch_results = _run_ethogram_batch(
                batch_indices=batch_indices,
                frame_map=frame_map,
                det_by_frame=det_by_frame,
                behavior_classes=behavior_classes,
                subject_desc=subject_desc,
                context_notes=context_notes,
                recent_labels=recent_labels[-MAX_CONTEXT_LABELS:],
                vlm_url=vlm_url,
                vlm_model=vlm_model,
            )

            # Calibrate from actual usage
            if batch_results.get("prompt_tokens", 0) > 0 and len(batch_indices) > 0:
                est_tokens_per_image = int(
                    0.3 * est_tokens_per_image
                    + 0.7 * max(500, batch_results["prompt_tokens"] / len(batch_indices))
                )
            cumulative_tokens += batch_results.get("total_tokens", 0)

            for lab in batch_results.get("labels", []):
                results.append(lab)
                recent_labels.append(lab)

                # Write to DB immediately (crash-safe)
                with _connect(run_dir) as conn:
                    conn.execute(
                        "INSERT INTO ethogram_labels "
                        "(frame_idx, behavior, confidence, note, profile_name) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (lab["frame_idx"], lab["behavior"],
                         lab.get("confidence", 0), lab.get("note", ""),
                         profile_name),
                    )
                    conn.commit()

            # Auto-compact: if cumulative tokens approach budget, reset context
            if cumulative_tokens > budget * 0.7:
                recent_labels = recent_labels[-3:]  # keep only last 3
                cumulative_tokens = 0

            idx += len(batch_indices)

        progress.empty()
        st.success(f"Coded {len(results)} frames → profile '{profile_name}'")
        existing = results

    if clear_col.button("🗑 Clear this profile"):
        with _connect(run_dir) as conn:
            n = conn.execute(
                "DELETE FROM ethogram_labels WHERE profile_name = ?",
                (profile_name,),
            ).rowcount
            conn.commit()
        st.info(f"Cleared {n} labels from profile '{profile_name}'")
        existing = []

    # ── display results ──
    if not existing:
        st.info(
            "No ethogram labels for this profile yet. "
            "Configure your behaviour classes and context, then click "
            "**Run ethogram analysis**."
        )
        return

    st.divider()
    st.header(f"Results — profile: {profile_name}")

    # Timeline
    _render_state_timeline(existing, frames, color_map)

    # Budget + Bout in two columns
    left, right = st.columns(2)
    with left:
        _render_behavior_budget(existing, color_map)
    with right:
        _render_transition_matrix(existing, frames)

    _render_bout_analysis(existing, frames)

    # Per-frame table
    st.subheader("Per-frame labels")
    frame_ts = {f["frame_idx"]: f["timestamp_ms"] / 1000.0 for f in frames}
    table_rows = []
    for lab in existing:
        ts = frame_ts.get(lab["frame_idx"], 0.0)
        table_rows.append({
            "frame": lab["frame_idx"],
            "t (s)": round(ts, 2),
            "behaviour": lab["behavior"],
            "confidence": round(lab.get("confidence", 0) or 0, 2),
            "note": lab.get("note", ""),
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Export
    st.subheader("Export")
    export_data = json.dumps(table_rows, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇ Download ethogram (JSON)",
        data=export_data,
        file_name=f"ethogram_{profile_name}.json",
        mime="application/json",
    )

    # CSV export
    import csv
    import io
    csv_buf = io.StringIO()
    if table_rows:
        writer = csv.DictWriter(csv_buf, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    st.download_button(
        "⬇ Download ethogram (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"ethogram_{profile_name}.csv",
        mime="text/csv",
    )


main()
