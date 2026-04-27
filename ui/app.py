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

    st.sidebar.subheader("Object Detection (YOLO26)")
    yolo_enabled = st.sidebar.checkbox("Enable", value=True, key="yolo_en")
    yolo_fps = st.sidebar.slider("Sweep FPS", 0.25, 4.0, 1.0, 0.25)
    yolo_size = st.sidebar.selectbox(
        "Model size",
        options=["n", "s", "m", "l", "x"],
        index=0,
        format_func=lambda s: {
            "n": "n — nano (~5 MB, fastest)",
            "s": "s — small (~22 MB)",
            "m": "m — medium (~50 MB)",
            "l": "l — large (~85 MB)",
            "x": "x — xlarge (~140 MB, most accurate)",
        }[s],
        help="Larger = more accurate but more VRAM and slower. "
             "On the 6 GB 3060, n/s are safe; m fits; l/x compete with "
             "depth model — use depth=off if you go big.",
    )
    use_seg = st.sidebar.checkbox(
        "Use segmentation variant",
        value=True,
        help="Seg variant emits the same bboxes plus instance masks "
             "(masks aren't persisted yet — bboxes are identical between "
             "yolo26<size>.pt and yolo26<size>-seg.pt).",
    )
    confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.subheader("Depth Estimation")
    depth_enabled = st.sidebar.checkbox(
        "Enable (spatial awareness — slower)", value=False
    )

    st.sidebar.subheader("Audio (whisper.cpp)")
    audio_enabled = st.sidebar.checkbox("Transcribe audio", value=True)
    whisper_model = st.sidebar.selectbox(
        "Whisper model",
        options=[
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3", "large-v3-turbo",
        ],
        index=3,  # base.en
        disabled=not audio_enabled,
        format_func=lambda m: {
            "tiny": "tiny — multilingual, ~75 MB, ~32× rt",
            "tiny.en": "tiny.en — English-only, ~75 MB, ~32× rt",
            "base": "base — multilingual, ~141 MB, ~16× rt",
            "base.en": "base.en — English-only, ~141 MB, ~16× rt (default)",
            "small": "small — multilingual, ~466 MB, ~6× rt",
            "small.en": "small.en — English-only, ~466 MB, ~6× rt",
            "medium": "medium — multilingual, ~1.5 GB, ~2× rt",
            "medium.en": "medium.en — English-only, ~1.5 GB, ~2× rt",
            "large-v1": "large-v1 — multilingual, ~3 GB, ~1× rt",
            "large-v2": "large-v2 — multilingual, ~3 GB, ~1× rt",
            "large-v3": "large-v3 — multilingual, ~3 GB, ~1× rt (best)",
            "large-v3-turbo": "large-v3-turbo — multilingual, ~1.6 GB, ~4× rt",
        }[m],
        help="Models auto-download from Hugging Face on first use "
             "to /home/ashie/whisper.cpp/models/. Speed multipliers are "
             "rough realtime ratios on a 3060 over Vulkan.",
    )
    whisper_device = st.sidebar.radio(
        "Whisper GPU (Vulkan device)",
        options=[1, 0, -1],
        index=0,
        format_func=lambda d: {1: "NVIDIA 3060 (recommended)",
                                0: "AMD 9070 XT",
                                -1: "CPU only"}[d],
        disabled=not audio_enabled,
        help="Vulkan device id. AMD radv occasionally throws device-lost "
             "errors; default is NVIDIA.",
    )

    st.sidebar.subheader("VLM Captioning (llama-server)")
    vlm_frame_mode = st.sidebar.radio(
        "Frame source for VLM",
        options=["keyframes", "scored"],
        index=0,
        format_func=lambda m: {
            "keyframes": "Keyframes only (motion-detected)",
            "scored":    "Top-K scored at Caption FPS",
        }[m],
        help="keyframes: send only the motion-detected keyframes from "
             "FrameSelector — no extra sampling. scored: pick top-K from all "
             "sampled frames at Caption FPS using FrameScorer (detection "
             "density + track churn + IoU change).",
    )
    vlm_fps = st.sidebar.slider(
        "Caption FPS", 0.1, 2.0, 0.5, 0.1,
        disabled=(vlm_frame_mode == "keyframes"),
        help="Ignored in keyframes-only mode.",
    )
    chunk_size = st.sidebar.slider("Frames per chunk", 4, 16, 10, 1)
    keep_vlm_running = st.sidebar.checkbox(
        "Keep VLM running after this video",
        value=False,
        help="Off (default): stop qwen35-turboquant.service when the run ends "
             "to free GPU VRAM. On: leave it loaded for back-to-back runs "
             "(saves ~10–15s startup per subsequent run).",
    )

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
            yolo_size=yolo_size,
            threshold_mult=threshold_mult,
            keep_vlm_running=keep_vlm_running,
            vlm_frame_mode=vlm_frame_mode,
            audio_enabled=audio_enabled,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
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
    yolo_size,
    threshold_mult,
    keep_vlm_running,
    vlm_frame_mode,
    audio_enabled,
    whisper_model,
    whisper_device,
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
    # Pre-compute ETA before the run starts (UI-only — pipeline doesn't see this)
    from src.runtime_estimator import estimate_run, format_seconds
    try:
        est = estimate_run(
            video_path=video_path,
            yolo_fps=yolo_fps,
            vlm_fps=vlm_fps,
            chunk_size=chunk_size,
            yolo_size=yolo_size,
            use_segmentation=use_seg,
            depth_enabled=depth_enabled,
            audio_enabled=audio_enabled,
            whisper_model=whisper_model,
            vlm_frame_mode=vlm_frame_mode,
            vlm_autostart_cold=not keep_vlm_running,
        )
    except Exception:
        est = None

    if est is not None:
        eta_total_s = int(est.total_s)
        eta_html = f"""
<div style="border:1px solid #2a2e35;border-radius:6px;padding:10px;
            background:#1c1f24;color:#e0e0e0;font-family:system-ui;font-size:14px">
  <div style="display:flex;justify-content:space-between;gap:14px;align-items:center">
    <div>
      <span style="color:#888">elapsed</span>
      <b id="elapsed" style="font-size:22px;color:#9ecbff">0:00</b>
      &nbsp;/&nbsp;
      <span style="color:#888">ETA</span>
      <b style="font-size:18px">{format_seconds(eta_total_s)}</b>
    </div>
    <div style="flex:1;max-width:400px">
      <div style="background:#0e1117;border-radius:4px;height:8px;overflow:hidden">
        <div id="bar" style="background:#42a5f5;height:8px;width:0%;
                              transition:width 0.4s linear"></div>
      </div>
    </div>
    <div style="color:#888;font-size:12px">
      <span id="pct">0%</span>
    </div>
  </div>
  <details style="margin-top:8px;color:#bbb">
    <summary style="cursor:pointer;color:#9ecbff">stage cost breakdown</summary>
    <ul style="margin:6px 0 0 20px;padding:0;font-size:12px;line-height:1.5">
      {''.join(f'<li><b>{s.name}</b> — {format_seconds(s.seconds)} &middot; <span style="color:#888">{s.note}</span></li>' for s in est.stages)}
    </ul>
  </details>
</div>
<script>
  (function() {{
    const t0 = Date.now();
    const total = {eta_total_s};
    const elapsedEl = document.getElementById('elapsed');
    const barEl = document.getElementById('bar');
    const pctEl = document.getElementById('pct');
    function fmt(s) {{
      s = Math.max(0, Math.floor(s));
      const m = Math.floor(s/60), sec = s%60;
      if (m >= 60) {{
        const h = Math.floor(m/60), mm = m%60;
        return h + ':' + String(mm).padStart(2,'0') + ':' + String(sec).padStart(2,'0');
      }}
      return m + ':' + String(sec).padStart(2,'0');
    }}
    setInterval(() => {{
      const e = (Date.now() - t0) / 1000;
      elapsedEl.textContent = fmt(e);
      const pct = total > 0 ? Math.min(100, (e/total)*100) : 0;
      barEl.style.width = pct.toFixed(1) + '%';
      pctEl.textContent = pct.toFixed(0) + '%';
      if (pct >= 100) {{
        barEl.style.background = '#ffa726';
        pctEl.textContent = '> ETA';
      }}
    }}, 200);
  }})();
</script>"""
        st.components.v1.html(eta_html, height=150)

    # Live VLM context-window panel — appears once Stage 5 starts emitting.
    ctx_panel = st.empty()
    ctx_state: dict = {"chunks": [], "n_ctx": None, "max_prompt": 0, "agg": None}

    def _render_ctx_panel() -> None:
        if not ctx_state["chunks"] and ctx_state["agg"] is None:
            return
        n_ctx = ctx_state["n_ctx"] or 100096
        rows_html = ""
        for c in ctx_state["chunks"]:
            pct = 100.0 * c["prompt_tokens"] / n_ctx if n_ctx else 0
            bar_color = "#42a5f5" if pct < 60 else "#ffa726" if pct < 85 else "#ef5350"
            rows_html += (
                f'<div style="display:flex;gap:10px;align-items:center;'
                f'font-size:12px;margin:2px 0">'
                f'<span style="color:#888;width:80px">chunk {c["chunk_id"]}/{c["n_total"]-1}</span>'
                f'<span style="color:#bbb;width:90px">{c["n_images"]} imgs</span>'
                f'<div style="flex:1;background:#0e1117;border-radius:3px;height:6px;'
                f'overflow:hidden">'
                f'<div style="background:{bar_color};height:6px;'
                f'width:{min(100,pct):.1f}%"></div></div>'
                f'<span style="color:#ddd;width:140px;text-align:right">'
                f'{c["prompt_tokens"]:,} / {n_ctx:,} ({pct:.1f}%)</span>'
                f'<span style="color:#888;width:60px;text-align:right">'
                f'{c["elapsed_s"]}s</span>'
                f'</div>'
            )
        agg_html = ""
        if ctx_state["agg"] is not None:
            a = ctx_state["agg"]
            pct = 100.0 * a["prompt_tokens"] / n_ctx if n_ctx else 0
            bar_color = "#42a5f5" if pct < 60 else "#ffa726" if pct < 85 else "#ef5350"
            agg_html = (
                f'<div style="display:flex;gap:10px;align-items:center;'
                f'font-size:12px;margin-top:6px;border-top:1px solid #2a2e35;padding-top:6px">'
                f'<span style="color:#888;width:80px">aggregator</span>'
                f'<span style="color:#bbb;width:90px">text-only</span>'
                f'<div style="flex:1;background:#0e1117;border-radius:3px;height:6px;'
                f'overflow:hidden">'
                f'<div style="background:{bar_color};height:6px;'
                f'width:{min(100,pct):.1f}%"></div></div>'
                f'<span style="color:#ddd;width:140px;text-align:right">'
                f'{a["prompt_tokens"]:,} / {n_ctx:,} ({pct:.1f}%)</span>'
                f'<span style="color:#888;width:60px;text-align:right"></span>'
                f'</div>'
            )
        max_pct = 100.0 * ctx_state["max_prompt"] / n_ctx if n_ctx else 0
        max_color = "#42a5f5" if max_pct < 60 else "#ffa726" if max_pct < 85 else "#ef5350"
        ctx_panel.markdown(
            f'<div style="border:1px solid #2a2e35;border-radius:6px;padding:10px;'
            f'background:#1c1f24;color:#e0e0e0;font-family:system-ui">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:6px">'
            f'<b style="font-size:13px">VLM context window usage (live)</b>'
            f'<span style="font-size:12px;color:{max_color}">'
            f'peak: {ctx_state["max_prompt"]:,} / {n_ctx:,} ({max_pct:.1f}%)</span>'
            f'</div>'
            f'{rows_html}{agg_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.status("Running v2 pipeline...", expanded=True) as status:
        try:
            stage_log = st.empty()
            stage_state: dict = {"lines": []}

            def _icon(event: str) -> str:
                return {
                    "start": "▶",
                    "done": "✓",
                    "skipped": "·",
                    "error": "✗",
                }.get(event, "•")

            def on_stage(name: str, event: str, info: dict) -> None:
                # Per-chunk VLM events update the context-window panel only —
                # they'd flood the stage log otherwise.
                if name == "VLM chunk" and event == "done":
                    if info.get("n_ctx"):
                        ctx_state["n_ctx"] = info["n_ctx"]
                    ctx_state["chunks"].append(info)
                    ctx_state["max_prompt"] = max(
                        ctx_state["max_prompt"], info.get("prompt_tokens", 0)
                    )
                    _render_ctx_panel()
                    return
                if name == "Aggregator call" and event == "done":
                    ctx_state["agg"] = info
                    ctx_state["max_prompt"] = max(
                        ctx_state["max_prompt"], info.get("prompt_tokens", 0)
                    )
                    _render_ctx_panel()
                    return

                if event == "start":
                    line = f"{_icon(event)} {name} — running…"
                    if info:
                        bits = ", ".join(f"{k}={v}" for k, v in info.items())
                        line += f"  ({bits})"
                    stage_state["lines"].append(line)
                else:
                    bits = ", ".join(f"{k}={v}" for k, v in info.items()) if info else ""
                    suffix = f"  ({bits})" if bits else ""
                    if stage_state["lines"] and stage_state["lines"][-1].startswith(
                        f"{_icon('start')} {name} —"
                    ):
                        stage_state["lines"][-1] = (
                            f"{_icon(event)} {name} — {event}{suffix}"
                        )
                    else:
                        stage_state["lines"].append(
                            f"{_icon(event)} {name} — {event}{suffix}"
                        )
                status.update(label=f"{name} ({event})")
                stage_log.code("\n".join(stage_state["lines"]), language="text")

            pipe = PipelineV2(
                config,
                camera_mode=camera_mode,
                yolo_fps=yolo_fps,
                vlm_fps=vlm_fps,
                chunk_size=chunk_size,
                depth_enabled=depth_enabled,
                use_segmentation=use_seg,
                yolo_size=yolo_size,
                threshold_mult=threshold_mult,
                vlm_autostart_service="qwen35-turboquant.service",
                vlm_autostop=not keep_vlm_running,
                vlm_frame_mode=vlm_frame_mode,
                audio_enabled=audio_enabled,
                whisper_model=whisper_model,
                whisper_gpu_device=(None if whisper_device == -1 else whisper_device),
                on_stage=on_stage,
            )
            result = pipe.run()
            status.update(label="Done", state="complete")
            actual_s = result.processing_time
            if est is not None:
                pct_of_eta = 100 * actual_s / max(1.0, est.total_s)
                delta_s = actual_s - est.total_s
                sign = "+" if delta_s >= 0 else "−"
                st.success(
                    f"Done in {format_seconds(actual_s)} "
                    f"(ETA was {format_seconds(est.total_s)}, "
                    f"{sign}{format_seconds(abs(delta_s))} = {pct_of_eta:.0f}% of ETA)"
                )
            else:
                st.success(f"Done in {actual_s:.1f}s")
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
