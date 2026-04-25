import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import plotly.graph_objects as go
import streamlit as st

from src.models import CameraMode, PipelineConfig
from src.modules.frame_selector import FrameSelector
from src.pipeline_v2 import PipelineV2


def main():
    st.set_page_config(page_title="TempoGraph v2", layout="wide")
    st.title("TempoGraph v2 - Chunked VLM Pipeline")

    st.sidebar.header("Pipeline Configuration")

    camera_mode_label = st.sidebar.radio(
        "Camera type",
        options=["Static / fixed (CCTV)", "Moving / handheld", "Auto-detect"],
        index=0,
    )
    camera_mode = {
        "Static / fixed (CCTV)": CameraMode.STATIC,
        "Moving / handheld": CameraMode.MOVING,
        "Auto-detect": CameraMode.AUTO,
    }[camera_mode_label]

    st.sidebar.subheader("Object Detection (YOLO)")
    yolo_enabled = st.sidebar.checkbox("Enable", value=True, key="yolo_en")
    yolo_fps = st.sidebar.slider("Sweep FPS", 0.25, 4.0, 1.0, 0.25)
    use_seg = st.sidebar.checkbox("Use segmentation variant (yolo11n-seg)", value=False)
    confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.subheader("Depth Estimation")
    depth_enabled = st.sidebar.checkbox(
        "Enable (spatial awareness — slower)", value=False
    )

    st.sidebar.subheader("VLM Captioning (llama-server)")
    vlm_fps = st.sidebar.slider("Caption FPS", 0.1, 2.0, 0.5, 0.1)
    chunk_size = st.sidebar.slider("Frames per chunk", 4, 16, 10, 1)

    st.sidebar.subheader("Frame Selection")
    threshold_mult = st.sidebar.slider("Keyframe threshold (× σ)", 0.5, 3.0, 1.0, 0.1)

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(uploaded.read())
        video_path = f.name

    col1, col2 = st.columns(2)
    if col1.button("Preview frame selection"):
        _render_selection_preview(video_path, camera_mode, yolo_fps, threshold_mult)

    if col2.button("Run full pipeline", type="primary"):
        _run_pipeline(
            video_path=video_path,
            video_name=uploaded.name,
            camera_mode=camera_mode,
            yolo_fps=yolo_fps,
            vlm_fps=vlm_fps,
            chunk_size=chunk_size,
            confidence=confidence,
            depth_enabled=depth_enabled,
            use_seg=use_seg,
            threshold_mult=threshold_mult,
        )


def _render_selection_preview(video_path, camera_mode, sample_fps, threshold_mult):
    selector = FrameSelector()
    result = selector.select(
        video_path=video_path,
        camera_mode=camera_mode,
        sample_fps=sample_fps,
        threshold_mult=threshold_mult,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.scan_indices,
            y=result.deltas,
            mode="lines",
            name="delta",
            line=dict(color="lightgray"),
        )
    )
    fig.add_hline(
        y=result.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"threshold={result.threshold:.2f}",
    )

    kf_set = set(result.keyframe_indices)
    smp_set = set(result.sampled_indices) - kf_set

    kf_x = [i for i in result.scan_indices if i in kf_set]
    kf_y = [
        result.deltas[result.scan_indices.index(i)] for i in kf_x
    ]
    fig.add_trace(
        go.Scatter(x=kf_x, y=kf_y, mode="markers", name="keyframes (mandatory)",
                   marker=dict(color="green", size=9))
    )

    smp_x = [i for i in result.scan_indices if i in smp_set]
    smp_y = [result.deltas[result.scan_indices.index(i)] for i in smp_x]
    fig.add_trace(
        go.Scatter(x=smp_x, y=smp_y, mode="markers", name=f"sampled @ {sample_fps} Hz",
                   marker=dict(color="orange", size=8))
    )
    fig.update_layout(
        title=f"Frame selection preview ({result.camera_mode.value})",
        xaxis_title="frame index",
        yaxis_title="delta",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Total frames to process: {len(result.frame_indices)} "
        f"({len(result.keyframe_indices)} green + {len(result.sampled_indices)} orange, "
        f"after dedup)"
    )


def _run_pipeline(
    video_path,
    video_name,
    camera_mode,
    yolo_fps,
    vlm_fps,
    chunk_size,
    confidence,
    depth_enabled,
    use_seg,
    threshold_mult,
):
    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": depth_enabled, "audio": False},
        fps=yolo_fps,
        max_frames=999,
        confidence=confidence,
        video_path=video_path,
        output_dir=f"results/{video_name}",
    )
    with st.status("Running v2 pipeline...", expanded=True) as status:
        try:
            pipe = PipelineV2(
                config,
                camera_mode=camera_mode,
                yolo_fps=yolo_fps,
                vlm_fps=vlm_fps,
                chunk_size=chunk_size,
                depth_enabled=depth_enabled,
                use_segmentation=use_seg,
                threshold_mult=threshold_mult,
            )
            result = pipe.run()
            status.update(label="Done", state="complete")
            st.success(f"Done in {result.processing_time:.1f}s")
            if result.analysis:
                st.subheader("Summary")
                st.write(result.analysis.summary or "(empty)")
                st.subheader("Entities")
                st.dataframe([
                    {"id": e.id, "type": e.type, "description": e.description}
                    for e in result.analysis.entities
                ])
                st.subheader("Visual events")
                st.dataframe([
                    {"type": e.type.value if hasattr(e.type, "value") else str(e.type),
                     "entities": ", ".join(e.entities), "start": e.start_time,
                     "end": e.end_time, "description": e.description}
                    for e in result.analysis.visual_events
                ])
        except Exception as e:
            status.update(label=f"Failed: {e}", state="error")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
