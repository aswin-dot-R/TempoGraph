import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import json
import os
import tempfile
from typing import List, Dict, Any

from src.models import PipelineConfig, PipelineResult, AnalysisResult
from src.pipeline import Pipeline

# Streamlit app
def main():
    st.set_page_config(page_title="TempoGraph", layout="wide")

    st.title("🎬 TempoGraph - Video Intelligence Platform")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    # Backend selector
    backend = st.sidebar.radio(
        "Backend",
        options=["Local (Ollama - qwen3-vl:4b)", "Cloud (Gemini Flash)"],
        index=0,
        help="Select analysis backend"
    )

    # Module selectors
    st.sidebar.subheader("Modules")
    behavior = st.sidebar.checkbox("Behavior Analysis", value=True)
    detection = st.sidebar.checkbox("Object Detection", value=True)
    depth = st.sidebar.checkbox("Depth Estimation", value=False)
    audio = st.sidebar.checkbox("Audio Analysis", value=True)

    # Processing parameters
    st.sidebar.subheader("Processing")
    fps = st.sidebar.slider("Frames Per Second", 0.5, 5.0, 1.0, 0.5)
    max_frames = st.sidebar.slider("Maximum Frames", 10, 120, 60, 10)
    confidence = st.sidebar.slider("Minimum Confidence", 0.1, 0.9, 0.5, 0.1)

    # Advanced settings expander
    with st.sidebar.expander("Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            resize_width = st.number_input(
                "Resize Width",
                min_value=1,
                max_value=1920,
                value=640,
                step=64
            )
        with col2:
            brightness = st.slider(
                "Brightness",
                min_value=-50,
                max_value=50,
                value=0,
                step=1
            )

    # Main interface
    st.header("📹 Upload Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        # Display video info
        video_name = uploaded_video.name
        st.info(f"Selected video: **{video_name}**")

        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(uploaded_video.read())
            video_path = f.name

        # Mock processing placeholder
        if st.button("▶ Analyze Video", type="primary"):
            with st.status("Running analysis pipeline...", expanded=True) as status:
                try:
                    # Create backend config
                    backend_mode = "llama-server" if "Local" in backend else "gemini"

                    # Check backend availability
                    if "Local" in backend:
                        import torch
                        if not torch.cuda.is_available():
                            status.error("❌ No GPU available for local backend. Please use Cloud (Gemini) mode.")
                            st.stop()

                    # Create pipeline config
                    config = PipelineConfig(
                        backend=backend_mode,
                        modules={
                            "behavior": behavior,
                            "detection": detection,
                            "depth": depth,
                            "audio": audio
                        },
                        fps=fps,
                        max_frames=max_frames,
                        confidence=confidence,
                        video_path=video_path,
                        output_dir=f"results/{video_name}"
                    )

                    # Run pipeline
                    status.write("Initializing pipeline...")
                    pipeline = Pipeline(config)

                    status.write("Extracting frames...")
                    result = pipeline.run()

                    status.update(label="Analysis complete!", state="complete", expanded=False)

                    st.success("Analysis completed!")

                    # Tabbed results
                    tabs = st.tabs([
                        "📹 Annotated Video",
                        "📊 Timeline",
                        "🔗 Interaction Graph",
                        "📝 Summary",
                        "📥 Export"
                    ])

                    # Tab 1: Annotated Video
                    with tabs[0]:
                        if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                            st.video(result.annotated_video_path)
                        else:
                            st.info("No annotated video available (detection/depth modules disabled)")

                    # Tab 2: Timeline
                    with tabs[1]:
                        if result.analysis:
                            # Create timeline data from analysis
                            timeline_data = {
                                "Entity": [],
                                "Event Type": [],
                                "Start Time": [],
                                "End Time": [],
                                "Description": [],
                                "Confidence": []
                            }

                            # Add visual events
                            for event in result.analysis.visual_events:
                                timeline_data["Entity"].extend(event.entities)
                                timeline_data["Event Type"].append(event.type.value)
                                timeline_data["Start Time"].append(event.start_time)
                                timeline_data["End Time"].append(event.end_time)
                                timeline_data["Description"].append(event.description)
                                timeline_data["Confidence"].append(event.confidence)

                            # Add audio events
                            for event in result.analysis.audio_events:
                                timeline_data["Entity"].append(event.speaker or "Audio")
                                timeline_data["Event Type"].append(event.type.value)
                                timeline_data["Start Time"].append(event.start_time)
                                timeline_data["End Time"].append(event.end_time)
                                timeline_data["Description"].append(event.text or event.label or "")
                                timeline_data["Confidence"].append(event.confidence)

                            if timeline_data["Entity"]:
                                # Convert to seconds for plotly
                                timeline_data["Start Sec"] = [
                                    _timestamp_to_seconds(ts) for ts in timeline_data["Start Time"]
                                ]
                                timeline_data["End Sec"] = [
                                    _timestamp_to_seconds(ts) for ts in timeline_data["End Time"]
                                ]

                                # Create plotly figure
                                fig = px.bar(
                                    timeline_data,
                                    x="Start Sec",
                                    y="End Sec",
                                    color="Event Type",
                                    text="Description",
                                    title="Event Timeline",
                                    labels={"Start Sec": "Start Time (seconds)", "End Sec": "End Time (seconds)"}
                                )
                                fig.update_traces(textposition='outside')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No events found in analysis")

                    # Tab 3: Interaction Graph
                    with tabs[2]:
                        st.subheader("Interaction Graph")
                        if result.analysis:
                            st.write(f"Entities: {len(result.analysis.entities)}")
                            st.write(f"Visual Events: {len(result.analysis.visual_events)}")
                            st.write(f"Audio Events: {len(result.analysis.audio_events)}")

                            # Simple graph visualization using networkx
                            try:
                                import networkx as nx
                                import matplotlib.pyplot as plt

                                G = nx.DiGraph()

                                # Add nodes for entities
                                for entity in result.analysis.entities:
                                    G.add_node(entity.id, type=entity.type, label=entity.description)

                                # Add edges for visual events
                                for i, event in enumerate(result.analysis.visual_events):
                                    for entity_id in event.entities:
                                        G.add_edge(entity_id, f"event_{i}", label=event.type.value)

                                # Draw graph
                                plt.figure(figsize=(12, 8))
                                pos = nx.spring_layout(G, k=0.5, iterations=50)

                                # Draw nodes
                                node_colors = []
                                for node in G.nodes():
                                    if node.startswith("event_"):
                                        node_colors.append("lightgray")
                                    else:
                                        entity = next((e for e in result.analysis.entities if e.id == node), None)
                                        if entity:
                                            node_colors.append(_get_color_for_type(entity.type))

                                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
                                nx.draw_networkx_labels(G, pos, font_size=8)
                                nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
                                nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))

                                plt.title("Entity Interaction Graph")
                                plt.axis('off')
                                st.pyplot(plt)
                            except ImportError:
                                st.info("Install networkx and matplotlib for graph visualization: pip install networkx matplotlib")
                                st.write("Graph would show entity interactions based on visual events")

                    # Tab 4: Summary
                    with tabs[3]:
                        st.subheader("Analysis Summary")
                        if result.analysis:
                            st.write(result.analysis.summary)

                            # Stats table
                            stats = {
                                "Total Entities": len(result.analysis.entities),
                                "Visual Events": len(result.analysis.visual_events),
                                "Audio Events": len(result.analysis.audio_events),
                                "Multimodal Correlations": len(result.analysis.multimodal_correlations),
                                "Processing Time": f"{result.processing_time:.1f}s"
                            }

                            st.table(stats)

                            # Entity list
                            if result.analysis.entities:
                                st.subheader("Entities")
                                entity_data = {
                                    "ID": [e.id for e in result.analysis.entities],
                                    "Type": [e.type for e in result.analysis.entities],
                                    "Description": [e.description for e in result.analysis.entities],
                                    "First Seen": [e.first_seen for e in result.analysis.entities],
                                    "Last Seen": [e.last_seen for e in result.analysis.entities]
                                }
                                st.dataframe(entity_data)

                            # Visual events list
                            if result.analysis.visual_events:
                                st.subheader("Visual Events")
                                event_data = {
                                    "Type": [e.type.value for e in result.analysis.visual_events],
                                    "Entities": [", ".join(e.entities) for e in result.analysis.visual_events],
                                    "Start": [e.start_time for e in result.analysis.visual_events],
                                    "End": [e.end_time for e in result.analysis.visual_events],
                                    "Description": [e.description for e in result.analysis.visual_events],
                                    "Confidence": [e.confidence for e in result.analysis.visual_events]
                                }
                                st.dataframe(event_data)

                            # Audio events list
                            if result.analysis.audio_events:
                                st.subheader("Audio Events")
                                audio_data = {
                                    "Type": [e.type.value for e in result.analysis.audio_events],
                                    "Speaker": [e.speaker or "N/A" for e in result.analysis.audio_events],
                                    "Text": [e.text or "N/A" for e in result.analysis.audio_events],
                                    "Start": [e.start_time for e in result.analysis.audio_events],
                                    "End": [e.end_time for e in result.analysis.audio_events],
                                    "Confidence": [e.confidence for e in result.analysis.audio_events]
                                }
                                st.dataframe(audio_data)

                    # Tab 5: Export
                    with tabs[4]:
                        st.subheader("Export Results")

                        # Create output directory
                        output_dir = Path(result.config.output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Save analysis JSON
                        analysis_json_path = output_dir / "analysis.json"
                        with open(analysis_json_path, "w") as f:
                            f.write(result.analysis.model_dump_json(indent=2))

                        st.download_button(
                            label="Download Analysis JSON",
                            data=open(analysis_json_path, "r").read(),
                            file_name="analysis.json",
                            mime="application/json"
                        )

                        # Save detection JSON
                        if result.detection:
                            detection_json_path = output_dir / "detection.json"
                            with open(detection_json_path, "w") as f:
                                f.write(result.detection.model_json(indent=2))

                            st.download_button(
                                label="Download Detection JSON",
                                data=open(detection_json_path, "r").read(),
                                file_name="detection.json",
                                mime="application/json"
                            )

                        # Save depth JSON
                        if result.depth:
                            depth_json_path = output_dir / "depth.json"
                            with open(depth_json_path, "w") as f:
                                f.write(result.depth.model_json(indent=2))

                            st.download_button(
                                label="Download Depth JSON",
                                data=open(depth_json_path, "r").read(),
                                file_name="depth.json",
                                mime="application/json"
                            )

                        # Download annotated video
                        if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                            with open(result.annotated_video_path, "rb") as f:
                                st.download_button(
                                    label="Download Annotated Video",
                                    data=f.read(),
                                    file_name="annotated.mp4",
                                    mime="video/mp4"
                                )

                except Exception as e:
                    status.error(f"❌ Analysis failed: {e}")
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **TempoGraph** is a video intelligence pipeline.
        - **Cloud mode**: Uses Gemini Flash API for video analysis
        - **Local mode**: Uses Qwen2.5-VL-3B with 4-bit quantization (requires 6GB+ GPU)
        """
    )

def _timestamp_to_seconds(ts: str) -> float:
    """Convert MM:SS timestamp to seconds."""
    try:
        minutes, seconds = ts.split(":")
        return float(minutes) * 60 + float(seconds)
    except:
        return 0.0

def _get_color_for_type(entity_type: str) -> str:
    """Get color for entity type."""
    colors = {
        "person": "#3498db",
        "dog": "#e74c3c",
        "cat": "#9b59b6",
        "vehicle": "#f39c12",
        "animal": "#2ecc71"
    }
    return colors.get(entity_type.lower(), "#95a5a6")

if __name__ == "__main__":
    main()