"""TempoGraph v2 — 'Drop → Plan → Run → Explore' UI.

Screen 1 (Landing): drop zone + path input + recent runs gallery.
Screen 2 (Plan): derived plan sentence + ETA + Adjust expander.
Screen 3 (Progress): stage checklist + Cancel + live counters.
After: auto-navigate to Results page.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

from ui.theme import apply_theme  # noqa: E402

# Ensure project root is on sys.path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_profile import VideoFacts, DerivedPlan, derive_plan, probe  # noqa: E402
from src.models import CameraMode, PipelineConfig  # noqa: E402
from src.pipeline_v2 import PipelineV2  # noqa: E402
from src.runtime_estimator import estimate_run, format_seconds  # noqa: E402

# ── state keys ───────────────────────────────────────────────────────
KEY_VIDEO = "tg_video_path"
KEY_NAME = "tg_video_name"
KEY_FACTS = "tg_video_facts"
KEY_PLAN = "tg_derived_plan"
KEY_OVERRIDES = "tg_override_values"
KEY_RUNNING = "tg_is_running"
KEY_CANCEL = "tg_cancel_event"

# ── legacy knob defaults ─────────────────────────────────────────────
DEFAULT_KNOBS = {
    "camera_mode": "auto",
    "yolo_enabled": True,
    "yolo_fps": 1.0,
    "yolo_size": "n",
    "yolo_seg": True,
    "confidence": 0.5,
    "depth_enabled": True,
    "audio_enabled": True,
    "whisper_model": "base.en",
    "whisper_device": 1,
    "skip_vlm": False,
    "vlm_frame_mode": "keyframes",
    "vlm_fps": 0.5,
    "chunk_size": 10,
    "vlm_dedup_threshold": 0.92,
    "dynamic_chunking": True,
    "context_threshold": 0.80,
    "keep_vlm_running": False,
    "threshold_mult": 1.0,
    "dense_captions": True,
}


def main():
    st.set_page_config(page_title="TempoGraph v2", layout="wide")
    apply_theme()
    st.markdown(
        '<div class="tg-wordmark">TEMPOGRAPH</div>'
        '<div class="tg-tagline">local video intelligence</div>',
        unsafe_allow_html=True,
    )
    st.title("TempoGraph v2")

    # ── Determine current screen ─────────────────────────────────────
    if KEY_RUNNING in st.session_state and st.session_state[KEY_RUNNING]:
        _render_progress_screen()
        return

    if KEY_PLAN not in st.session_state or st.session_state[KEY_PLAN] is None:
        _render_landing_screen()
        return

    _render_plan_screen()


# ── Screen 1: Landing ───────────────────────────────────────────────
def _render_landing_screen():
    # Zero sidebar widgets on landing
    st.sidebar.markdown("### Welcome")
    st.sidebar.caption("Drop a video or enter a file path to begin.")

    # ── Drop zone (hero) ───────────────────────────────────────────
    st.markdown(
        '<div class="tg-hero">'
        '<div class="tg-dropzone">'
        "<p style='color:#888;font-size:16px;margin-bottom:4px'>"
        "Drop a video</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "",
        type=["mp4", "avi", "mov", "mkv", "webm", "flv"],
        label_visibility="collapsed",
        key="landing_uploader",
    )

    # ── Path / URL input ─────────────────────────────────────────────
    path_input = st.text_input(
        "Or enter a local file path:",
        placeholder="/home/username/videos/clip.mp4",
        key="path_input",
    )

    # ── Resolve video ────────────────────────────────────────────────
    video_path: Optional[str] = None
    video_name: Optional[str] = None

    if uploaded is not None:
        video_name = uploaded.name
        ext = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(uploaded.read())
            video_path = f.name
    elif path_input and os.path.isfile(path_input):
        video_name = os.path.basename(path_input)
        video_path = path_input

    if video_path is None:
        st.session_state[KEY_PLAN] = None
        _render_recent_runs()
        return

    # ── Probe + derive plan ──────────────────────────────────────────
    try:
        facts = probe(video_path)
        plan = derive_plan(facts)
        st.session_state[KEY_FACTS] = facts
        st.session_state[KEY_PLAN] = plan
        st.session_state[KEY_VIDEO] = video_path
        st.session_state[KEY_NAME] = video_name
        st.session_state[KEY_OVERRIDES] = dict(DEFAULT_KNOBS)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to probe video: {e}")
        _render_recent_runs()


def _metric_card(label: str, value: int) -> None:
    """Render a single stat card for recent runs."""
    st.markdown(
        '<div class="tg-metric-card">'
        f'<div class="tg-metric-value">{value}</div>'
        f'<div class="tg-metric-label">{label}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_recent_runs():
    """Show recent runs gallery below the landing controls."""
    results_dir = Path("results")
    if not results_dir.exists():
        return

    runs = []
    for d in results_dir.iterdir():
        if d.is_dir() and (d / "tempograph.db").exists():
            try:
                with sqlite3.connect(str(d / "tempograph.db")) as conn:
                    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
                    n_dets = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[
                        0
                    ]
                n_entities = 0
                n_events = 0
                n_captions = 0
                analysis = d / "analysis.json"
                if analysis.exists():
                    data = json.loads(analysis.read_text())
                    n_entities = len(data.get("entities", []))
                    n_events = len(data.get("visual_events", []))
                n_captions = 0
                chunks = d / "chunks.json"
                if chunks.exists():
                    data = json.loads(chunks.read_text())
                    n_captions = sum(1 for c in data if c.get("summary"))

                from datetime import datetime

                mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%b %d")
                runs.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "mtime": mtime,
                        "n_frames": n_frames,
                        "n_detections": n_dets,
                        "n_entities": n_entities,
                        "n_events": n_events,
                        "n_captions": n_captions,
                    }
                )
            except Exception:
                continue

    if not runs:
        return

    st.divider()
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:8px">'
        '<span style="font-size:16px;font-weight:600;color:#E6EAEE">Recent runs</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    for r in runs[:8]:
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(5,1fr) 120px;gap:10px;align-items:start">',
            unsafe_allow_html=True,
        )
        _metric_card("Frames", r["n_frames"])
        _metric_card("Detections", r["n_detections"])
        _metric_card("Entities", r["n_entities"])
        _metric_card("Events", r["n_events"])
        st.markdown(
            '<div class="tg-metric-card">'
            f'<div class="tg-metric-label" style="color:#6a737a">{r["mtime"]}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button(
            f"Open: {r['name']}", key=f"open_run_{r['name']}", use_container_width=True
        ):
            st.session_state["selected_run"] = r["name"]
            st.switch_page("pages/Results.py")


# ── Screen 2: Plan ──────────────────────────────────────────────────
def _render_plan_screen():
    facts: VideoFacts = st.session_state[KEY_FACTS]
    plan: DerivedPlan = st.session_state[KEY_PLAN]
    video_path: str = st.session_state[KEY_VIDEO]
    video_name: str = st.session_state[KEY_NAME]

    # Compute ETA
    try:
        est = estimate_run(
            video_path=video_path,
            yolo_fps=plan.yolo_fps,
            vlm_fps=plan.vlm_fps,
            chunk_size=plan.chunk_size,
            yolo_size=plan.yolo_size,
            use_segmentation=plan.yolo_seg,
            depth_enabled=plan.depth_enabled,
            audio_enabled=plan.audio_enabled,
            whisper_model=plan.whisper_model,
            vlm_frame_mode=plan.vlm_frame_mode,
            vlm_autostart_cold=True,
        )
    except Exception:
        est = None

    eta_str = format_seconds(est.total_s) if est else "~?"
    plan_sentence = (
        f"Analyzing `{video_name}` ({facts.duration_s:.0f}s, "
        f"{facts.width}×{facts.height}, "
        f"{'audio' if facts.has_audio else 'silent'}) — "
        f"YOLO{plan.yolo_size}-seg @ {plan.yolo_fps} fps, "
        f"depth {'on' if plan.depth_enabled else 'off'}, "
        f"audio {'on' if plan.audio_enabled else 'off'}, "
        f"VLM keyframes @ {plan.vlm_fps} fps. "
        f"Estimated time: {eta_str}."
    )

    st.info(plan_sentence)

    col1, col2 = st.columns([1, 3])
    if col1.button(
        "Analyze", type="primary", use_container_width=True, key="analyze_btn"
    ):
        st.session_state[KEY_RUNNING] = True
        st.session_state[KEY_CANCEL] = threading.Event()
        st.rerun()

    # ── Adjust plan expander ─────────────────────────────────────────
    with st.expander("Adjust plan", expanded=False):
        _render_knobs(plan)

    # Show recent runs
    _render_recent_runs()


def _render_knobs(plan: DerivedPlan):
    """Render the legacy knobs, pre-filled from the derived plan."""
    overrides = st.session_state.get(KEY_OVERRIDES, dict(DEFAULT_KNOBS))

    # Camera mode
    camera_opts = ["auto", "static", "moving"]
    overrides["camera_mode"] = st.sidebar.selectbox(
        "Camera mode",
        camera_opts,
        index=camera_opts.index(overrides.get("camera_mode", "auto")),
        key="knob_camera_mode",
    )

    # YOLO section
    st.sidebar.subheader("Object Detection (YOLO26)")
    overrides["yolo_enabled"] = st.sidebar.checkbox(
        "Enable", value=overrides.get("yolo_enabled", True), key="knob_yolo_en"
    )
    overrides["yolo_fps"] = st.sidebar.slider(
        "Sweep FPS",
        0.25,
        4.0,
        overrides.get("yolo_fps", 1.0),
        0.25,
        key="knob_yolo_fps",
    )
    yolo_sizes = ["n", "s", "m", "l", "x"]
    overrides["yolo_size"] = st.sidebar.selectbox(
        "Model size",
        yolo_sizes,
        index=yolo_sizes.index(overrides.get("yolo_size", "n")),
        key="knob_yolo_size",
    )
    overrides["yolo_seg"] = st.sidebar.checkbox(
        "Segmentation variant",
        value=overrides.get("yolo_seg", True),
        key="knob_yolo_seg",
    )
    overrides["confidence"] = st.sidebar.slider(
        "Confidence", 0.1, 0.9, overrides.get("confidence", 0.5), 0.05, key="knob_conf"
    )

    # Depth
    st.sidebar.subheader("Depth Estimation")
    overrides["depth_enabled"] = st.sidebar.checkbox(
        "Enable", value=overrides.get("depth_enabled", True), key="knob_depth"
    )

    # Audio
    st.sidebar.subheader("Audio (whisper.cpp)")
    overrides["audio_enabled"] = st.sidebar.checkbox(
        "Transcribe audio", value=overrides.get("audio_enabled", True), key="knob_audio"
    )
    whisper_models = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    ]
    overrides["whisper_model"] = st.sidebar.selectbox(
        "Whisper model",
        whisper_models,
        index=whisper_models.index(overrides.get("whisper_model", "base.en")),
        key="knob_whisper_model",
        disabled=not overrides["audio_enabled"],
    )
    overrides["whisper_device"] = st.sidebar.radio(
        "GPU device",
        [1, 0, -1],
        index=[1, 0, -1].index(overrides.get("whisper_device", 1)),
        format_func=lambda d: {1: "NVIDIA", 0: "AMD", -1: "CPU"}[d],
        key="knob_whisper_dev",
        disabled=not overrides["audio_enabled"],
    )

    # VLM
    st.sidebar.subheader("VLM Captioning")
    overrides["skip_vlm"] = st.sidebar.checkbox(
        "Skip VLM (preprocess only)",
        value=overrides.get("skip_vlm", False),
        key="knob_skip_vlm",
    )
    overrides["vlm_frame_mode"] = st.sidebar.radio(
        "Frame source",
        ["keyframes", "scored"],
        index=["keyframes", "scored"].index(
            overrides.get("vlm_frame_mode", "keyframes")
        ),
        key="knob_vlm_mode",
        disabled=overrides["skip_vlm"],
    )
    overrides["vlm_fps"] = st.sidebar.slider(
        "Caption FPS",
        0.1,
        2.0,
        overrides.get("vlm_fps", 0.5),
        0.1,
        key="knob_vlm_fps",
        disabled=overrides["vlm_frame_mode"] == "keyframes" or overrides["skip_vlm"],
    )
    overrides["chunk_size"] = st.sidebar.slider(
        "Frames per chunk",
        4,
        16,
        overrides.get("chunk_size", 10),
        1,
        key="knob_chunk",
        disabled=overrides["skip_vlm"],
    )
    overrides["vlm_dedup_threshold"] = st.sidebar.slider(
        "Dedup threshold",
        0.0,
        1.0,
        overrides.get("vlm_dedup_threshold", 0.92),
        0.01,
        key="knob_dedup",
    )
    overrides["dynamic_chunking"] = st.sidebar.checkbox(
        "Dynamic chunking",
        value=overrides.get("dynamic_chunking", True),
        key="knob_dynamic_chunk",
    )
    overrides["context_threshold"] = st.sidebar.slider(
        "Context threshold",
        0.50,
        0.95,
        overrides.get("context_threshold", 0.80),
        0.05,
        key="knob_ctx_thresh",
        disabled=not overrides["dynamic_chunking"],
    )
    overrides["keep_vlm_running"] = st.sidebar.checkbox(
        "Keep VLM running",
        value=overrides.get("keep_vlm_running", False),
        key="knob_keep_vlm",
    )

    # Frame selection
    st.sidebar.subheader("Frame Selection")
    overrides["threshold_mult"] = st.sidebar.slider(
        "Keyframe threshold (× σ)",
        0.5,
        3.0,
        overrides.get("threshold_mult", 1.0),
        0.1,
        key="knob_thresh",
    )


# ── Screen 3: Progress ──────────────────────────────────────────────
def _render_progress_screen():
    video_path = st.session_state[KEY_VIDEO]
    video_name = st.session_state[KEY_NAME]
    overrides = st.session_state.get(KEY_OVERRIDES, dict(DEFAULT_KNOBS))
    cancel_event = st.session_state.get(KEY_CANCEL, threading.Event())

    if cancel_event.is_set():
        st.session_state[KEY_RUNNING] = False
        st.session_state[KEY_PLAN] = None
        st.warning("Pipeline cancelled.")
        st.rerun()

    st.set_page_config(page_title=f"Running: {video_name}", layout="wide")
    st.title(f"Analyzing: {video_name}")

    # Cancel button
    if st.button("Cancel", type="secondary"):
        cancel_event.set()
        st.rerun()

    # Build pipeline kwargs
    out_dir = f"results/{video_name}"
    config = PipelineConfig(
        backend="llama-server",
        modules={
            "behavior": True,
            "detection": True,
            "depth": overrides["depth_enabled"],
            "audio": overrides["audio_enabled"],
        },
        fps=overrides["yolo_fps"],
        max_frames=999,
        confidence=overrides["confidence"],
        video_path=video_path,
        output_dir=out_dir,
    )

    stage_state: dict = {
        "lines": [],
        "counts": {"entities": 0, "detections": 0, "captions": 0},
    }
    stage_status = {
        "Frame selection": "queued",
        "Audio transcription": "queued",
        "YOLO detection": "queued",
        "Dense captions": "queued",
        "Depth estimation": "queued",
        "Frame scoring": "queued",
        "VLM captioning": "queued",
        "Aggregation": "queued",
    }

    navigate_to_results = False
    with st.spinner("Running pipeline..."):
        try:
            camera_map = {
                "auto": CameraMode.AUTO,
                "static": CameraMode.STATIC,
                "moving": CameraMode.MOVING,
            }
            cam_mode = camera_map.get(overrides["camera_mode"], CameraMode.AUTO)

            pipe = PipelineV2(
                config,
                camera_mode=cam_mode,
                yolo_fps=overrides["yolo_fps"],
                vlm_fps=overrides["vlm_fps"],
                chunk_size=overrides["chunk_size"],
                depth_enabled=overrides["depth_enabled"],
                use_segmentation=overrides["yolo_seg"],
                yolo_size=overrides["yolo_size"],
                threshold_mult=overrides["threshold_mult"],
                skip_vlm=overrides["skip_vlm"],
                # Ornith 1.0 9B (vision) runs permanently on :8085 and also
                # serves the opencode coder — no autostart, NEVER autostop.
                # (qwen35-turboquant is retired; its old port 8082 is now the
                # free-claude-code proxy.)
                vlm_url="http://127.0.0.1:8085",
                vlm_model="ornith-1.0-9b-Q4_K_M.gguf",
                vlm_autostart_service=None,
                vlm_autostop=False,
                dense_captions=overrides.get("dense_captions", True),
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                vlm_frame_mode=overrides["vlm_frame_mode"],
                vlm_dedup_threshold=overrides["vlm_dedup_threshold"],
                dynamic_chunking=overrides["dynamic_chunking"],
                context_threshold=overrides["context_threshold"],
                audio_enabled=overrides["audio_enabled"],
                whisper_model=overrides["whisper_model"],
                whisper_gpu_device=(
                    None
                    if overrides["whisper_device"] == -1
                    else overrides["whisper_device"]
                ),
                cancel_event=cancel_event,
                on_stage=lambda name, event, info: _on_stage_progress(
                    name, event, info, stage_status, stage_state
                ),
            )
            result = pipe.run()

            # Update final counts
            try:
                import sqlite3

                db_path = Path(out_dir) / "tempograph.db"
                with sqlite3.connect(str(db_path)) as conn:
                    stage_state["counts"]["detections"] = conn.execute(
                        "SELECT COUNT(*) FROM detections"
                    ).fetchone()[0]
            except Exception:
                pass
            if result.analysis:
                stage_state["counts"]["entities"] = len(result.analysis.entities)
                stage_state["counts"]["captions"] = len(result.analysis.visual_events)

            # Mark all done
            for s in stage_status:
                stage_status[s] = "done"

            st.success(f"Done in {result.processing_time:.1f}s")

            # Navigate to results — outside the try below, because
            # st.switch_page signals via a control-flow exception that a
            # generic `except Exception` would swallow as a failure.
            st.session_state[KEY_RUNNING] = False
            st.session_state[KEY_PLAN] = None
            st.session_state["selected_run"] = video_name
            navigate_to_results = True

        except KeyboardInterrupt:
            st.warning("Pipeline cancelled.")
            st.session_state[KEY_RUNNING] = False
            st.session_state[KEY_PLAN] = None
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state[KEY_RUNNING] = False

        if navigate_to_results:
            st.info("Analysis complete. Opening results...")
            st.switch_page("pages/Results.py")

    # Stage checklist
    st.divider()
    st.subheader("Stage status")
    _glyph = {
        "queued": '<span style="color:#6a737a">&#x25ce;</span>',
        "running": '<span style="color:#3FBFB5">&#x25cf;</span>',
        "done": '<span style="color:#34d399">&#x2714;</span>',
        "error": '<span style="color:#f87171">&#x2715;</span>',
    }
    for stage, status in stage_status.items():
        glyph = _glyph.get(status, '<span style="color:#6a737a">•</span>')
        st.markdown(
            f'<div class="tg-stage">'
            f"{glyph} "
            f'<span class="tg-stage-name">{stage}</span> &mdash; '
            f'<span style="color:#888">{status}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Live counters
    c1, c2, c3 = st.columns(3)
    c1.metric("Entities detected", stage_state["counts"]["entities"])
    c2.metric("Detections", stage_state["counts"]["detections"])
    c3.metric("Captions", stage_state["counts"]["captions"])

    # Stage log
    if stage_state["lines"]:
        st.code("\n".join(stage_state["lines"]), language="text")


def _on_stage_progress(
    name: str, event: str, info: dict, stage_status: dict, stage_state: dict
) -> None:
    """Callback for pipeline stage events — renders the progress screen."""
    if name == "VLM chunk" and event == "done":
        return  # handled separately to avoid flooding

    if name == "Aggregator call" and event == "done":
        return

    # Update stage status
    if event == "start":
        stage_status[name] = "running"
    elif event == "done":
        stage_status[name] = "done"
        # Update counts
        if name == "YOLO detection" and "detections" in info:
            stage_state["counts"]["detections"] = info["detections"]
        if name == "Frame selection" and "frames" in info:
            stage_state["counts"]["frames"] = info["frames"]
    elif event == "skipped":
        stage_status[name] = "done"
    elif event == "error":
        stage_status[name] = "error"

    # Build log line
    icon = {"start": "▶", "done": "✓", "skipped": "·", "error": "✗"}.get(event, "•")
    bits = ", ".join(f"{k}={v}" for k, v in info.items()) if info else ""
    suffix = f"  ({bits})" if bits else ""
    line = f"{icon} {name} — {event}{suffix}"
    stage_state["lines"].append(line)


if __name__ == "__main__":
    main()
