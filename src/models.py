from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class BehaviorType(str, Enum):
    APPROACH = "approach"
    DEPART = "depart"
    INTERACT = "interact"
    FOLLOW = "follow"
    IDLE = "idle"
    GROUP = "group"
    AVOID = "avoid"
    CHASE = "chase"
    OBSERVE = "observe"
    MOVING = "moving"
    WALKING = "walking"
    RUNNING = "running"
    STANDING = "standing"
    SITTING = "sitting"
    PLAYING = "playing"
    JUMPING = "jumping"
    OTHER = "other"


class SoundType(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"
    ANIMAL_SOUND = "animal_sound"
    VEHICLE = "vehicle"
    IMPACT = "impact"
    ENVIRONMENTAL = "environmental"
    SILENCE = "silence"


class Entity(BaseModel):
    id: str
    type: str
    description: str
    first_seen: str  # MM:SS format
    last_seen: str


class VisualEvent(BaseModel):
    type: BehaviorType
    entities: List[str]  # entity IDs
    start_time: str
    end_time: str
    description: str
    confidence: float = 0.5


class AudioEvent(BaseModel):
    type: SoundType
    start_time: str
    end_time: str
    speaker: Optional[str] = None
    text: Optional[str] = None
    label: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.5


class MultimodalCorrelation(BaseModel):
    visual_event: int  # index into visual_events
    audio_event: int  # index into audio_events
    description: str


class AnalysisResult(BaseModel):
    entities: List[Entity] = []
    visual_events: List[VisualEvent] = []
    audio_events: List[AudioEvent] = []
    multimodal_correlations: List[MultimodalCorrelation] = []
    summary: str = ""


class DetectionBox(BaseModel):
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    track_id: Optional[int] = None


class DetectionResult(BaseModel):
    boxes: List[DetectionBox] = []
    fps: float = 1.0
    frame_count: int = 0


class DepthFrame(BaseModel):
    frame_idx: int
    depth_map_path: str  # path to saved .npy


class DepthResult(BaseModel):
    frames: List[DepthFrame] = []


class PipelineConfig(BaseModel):
    backend: str = "gemini"
    modules: dict = {"behavior": True, "detection": True, "depth": False, "audio": True}
    fps: float = 1.0
    max_frames: int = 60
    confidence: float = 0.5
    video_path: str = ""
    output_dir: str = "results"


class PipelineResult(BaseModel):
    analysis: Optional[AnalysisResult] = None
    detection: Optional[DetectionResult] = None
    depth: Optional[DepthResult] = None
    config: Optional[PipelineConfig] = None
    annotated_video_path: Optional[str] = None
    processing_time: float = 0.0
