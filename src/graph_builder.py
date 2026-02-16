import networkx as nx
import json
import logging
from typing import List, Dict, Optional, Any
from src.models import AnalysisResult, PipelineResult
from enum import Enum

# Entity and VisualEvent types are already defined in models.py
# We'll import the enums directly for clarity
from src.models import BehaviorType, SoundType

_BEHAVIOR_VALUES = {bt.value for bt in BehaviorType}


class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # directed, allows multiple edges
        self.logger = logging.getLogger(__name__)

    def build(self, result: AnalysisResult) -> nx.MultiDiGraph:
        """Build graph from analysis result."""
        self.graph.clear()

        # Add entity nodes
        for entity in result.entities:
            self.graph.add_node(
                entity.id,
                type=entity.type,
                description=entity.description,
                first_seen=entity.first_seen,
                last_seen=entity.last_seen,
                # visual attributes for rendering
                color=self._get_color_for_type(entity.type),
                size=30,
            )

        # Add visual event edges
        for i, event in enumerate(result.visual_events):
            if len(event.entities) >= 2:
                # Add edge between first two entities
                self.graph.add_edge(
                    event.entities[0],
                    event.entities[1],
                    key=f"ve_{i}",
                    type=event.type.value,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    description=event.description,
                    confidence=event.confidence,
                )
            elif len(event.entities) == 1:
                # Self-referencing event (idle, observe)
                # Store as node attribute
                pass

        # Store audio events as graph-level attributes
        self.graph.graph["audio_events"] = [e.model_dump() for e in result.audio_events]
        self.graph.graph["summary"] = result.summary
        self.graph.graph["correlations"] = [
            c.model_dump() for c in result.multimodal_correlations
        ]

        return self.graph

    def to_json(self) -> Dict[str, Any]:
        """Export graph to JSON for frontend."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "type": data.get("type"),
                    "description": data.get("description"),
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                    "color": data.get("color"),
                    "size": data.get("size"),
                }
            )

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "type": data.get("type"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "description": data.get("description"),
                    "confidence": data.get("confidence"),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "audio_events": self.graph.graph.get("audio_events", []),
            "summary": self.graph.graph.get("summary", ""),
            "correlations": self.graph.graph.get("correlations", []),
        }

    def to_pyvis_html(self, output_path: str = "graph.html") -> str:
        """Generate interactive pyvis HTML visualization."""
        try:
            from pyvis.network import Network
        except ImportError:
            self.logger.error("pyvis not installed for to_pyvis_html")
            return ""

        net = Network(height="600px", width="100%", notebook=False, directed=True)
        net.barnes_hut()

        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            net.add_node(
                node_id,
                label=node_id,
                title=f"{data.get('type', '')}: {data.get('description', '')}",
                color=data.get("color"),
                size=data.get("size", 30),
            )

        # Add edges
        for src, dst, data in self.graph.edges(data=True):
            net.add_edge(
                src,
                dst,
                label=data.get("type"),
                title=data.get("description", ""),
                color=self._get_edge_color(data.get("type")),
            )

        # Save HTML
        net.save_graph(output_path)
        return output_path

    def query_by_entities(self, e1: str, e2: str) -> List[Dict]:
        """Get all events between two entities."""
        results = []
        for _, _, data in self.graph.edges(data=True):
            if data.get("type") in _BEHAVIOR_VALUES:
                if data.get("start_time") and data.get("end_time"):
                    results.append(
                        {
                            "type": data.get("type"),
                            "start_time": data.get("start_time"),
                            "end_time": data.get("end_time"),
                            "description": data.get("description"),
                        }
                    )
        return results

    def query_by_time(self, timestamp: str) -> Dict[str, List]:
        """Get all events active at a given timestamp."""
        query_secs = self._normalize_timestamp(timestamp)

        active_events = []
        for _, _, data in self.graph.edges(data=True):
            start = data.get("start_time")
            end = data.get("end_time")
            if start and end:
                start_secs = self._normalize_timestamp(start)
                end_secs = self._normalize_timestamp(end)
                if start_secs <= query_secs <= end_secs:
                    active_events.append(
                        {
                            "type": data.get("type"),
                            "description": data.get("description"),
                            "confidence": data.get("confidence"),
                        }
                    )

        return {"events": active_events, "timestamp": query_secs}

    def get_timeline(self) -> List[Dict]:
        """Get all events sorted by start_time for timeline view."""
        timeline_events = []

        # Collect all visual events
        for event in self.graph.graph.get("visual_events", []):
            timeline_events.append(
                {
                    "type": event.type.value,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "description": event.description,
                    "entities": event.entities,
                    "confidence": event.confidence,
                }
            )

        # Collect all audio events
        audio_events = self.graph.graph.get("audio_events", [])
        for audio_event in audio_events:
            timeline_events.append(
                {
                    "type": "audio",
                    "start_time": audio_event.get("start_time"),
                    "end_time": audio_event.get("end_time"),
                    "description": audio_event.get("label"),
                    "audio_text": audio_event.get("text"),
                    "confidence": audio_event.get("confidence"),
                }
            )

        # Sort by start_time (convert to seconds for proper sorting)
        def time_to_seconds(ts):
            if ":" in ts:
                parts = ts.split(":")
                return int(parts[0]) * 60 + float(parts[1])
            return float(ts)

        timeline_events.sort(key=lambda x: time_to_seconds(x["start_time"]))
        return timeline_events

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        stats = {}

        # Total entities
        stats["total_entities"] = len(self.graph.nodes())

        # Total visual events (edges with behavior types)
        visual_edges = [
            edge
            for _, _, edge in self.graph.edges(data=True)
            if edge.get("type") in _BEHAVIOR_VALUES
        ]
        stats["total_visual_events"] = len(visual_edges)

        # Most active entity (most connections)
        node_connections = {}
        for node in self.graph.nodes():
            connections = self.graph.in_degree(node) + self.graph.out_degree(node)
            node_connections[node] = connections

        if node_connections:
            most_active = max(node_connections.items(), key=lambda x: x[1])
            stats["most_active_entity"] = most_active[0]
            stats["connection_count"] = most_active[1]

        # Dominant behavior type
        behavior_counts = {}
        for _, _, edge in self.graph.edges(data=True):
            if edge.get("type") in _BEHAVIOR_VALUES:
                behavior = edge.get("type")
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        if behavior_counts:
            dominant = max(behavior_counts.items(), key=lambda x: x[1])
            stats["dominant_behavior"] = dominant[0]
            stats["behavior_count"] = dominant[1]

        return stats

    def _get_color_for_type(self, entity_type: str) -> str:
        """Return hex color for entity type."""
        colors = {
            "person": "#4A90D9",
            "dog": "#E8943A",
            "cat": "#9B59B6",
            "vehicle": "#2ECC71",
            "object": "#95A5A6",
        }
        return colors.get(entity_type.lower(), "#BDC3C7")

    def _get_edge_color(self, behavior_type: str) -> str:
        """Return color for edge based on behavior type."""
        colors = {
            BehaviorType.APPROACH.value: "#4A90D9",
            BehaviorType.DEPART.value: "#95A5A6",
            BehaviorType.INTERACT.value: "#2ECC71",
            BehaviorType.FOLLOW.value: "#9B59B6",
            BehaviorType.IDLE.value: "#BDC3C7",
            BehaviorType.GROUP.value: "#E74C3C",
            BehaviorType.AVOID.value: "#34495E",
            BehaviorType.CHASE.value: "#F1C40F",
            BehaviorType.OBSERVE.value: "#7F8C8D",
        }
        return colors.get(behavior_type, "#7F8C8D")

    def _normalize_timestamp(self, ts: str) -> str:
        """Convert MM:SS to seconds."""
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(ts)
