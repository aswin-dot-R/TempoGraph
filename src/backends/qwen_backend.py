"""
Qwen3-4B local backend for TempoGraph.

CRITICAL VRAM NOTES:
- This model uses 4-bit quantization to fit in 6GB VRAM
- Load with BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
- After inference: del model, del processor, torch.cuda.empty_cache(), gc.collect()
- Peak VRAM should be ~2.5-3GB

EXACT API USAGE for Qwen3-VL:

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Build message with multiple frames as images:
messages = [
    {
        "role": "user",
        "content": [
            # Add each frame as an image
            {"type": "image", "image": f"file://{frame_path}"}
            for frame_path in frame_paths
        ] + [
            {"type": "text", "text": ANALYSIS_PROMPT_LOCAL}
        ]
    }
]

# Process with qwen_vl_utils
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
output_ids_trimmed = output_ids[0][inputs.input_ids.shape[1]:]
response_text = processor.decode(output_ids_trimmed, skip_special_tokens=True)

IMPORTANT CAVEATS:
- Qwen-VL can't process audio — only visual frames
- For local audio: separate Whisper pipeline handles that
- The local prompt should NOT reference audio analysis
- Limit to ~15-20 frames max to stay within context + VRAM budget
- If more frames than limit: subsample uniformly
- qwen_vl_utils is from the qwen-vl-utils pip package

LOCAL-ONLY PROMPT (no audio — Whisper handles that separately):
The prompt should ask for entities + visual_events only.
Audio events will be merged from Whisper separately.
"""

import gc
import torch
import logging
from pathlib import Path
from typing import List, Optional
from src.backends.base import BaseVLMBackend
from src.models import AnalysisResult
from src.json_parser import JSONParser

ANALYSIS_PROMPT_LOCAL = """You are a video analysis AI. Analyze these video frames
(sampled at regular intervals) and identify:

1. All distinct entities (people, animals, objects) with unique IDs
2. All temporal behaviors and interactions between entities
3. Estimated timestamps based on frame position

Output ONLY valid JSON:
{
  "entities": [
    {"id": "E1", "type": "dog", "description": "brown labrador",
     "first_seen": "00:02", "last_seen": "00:45"}
  ],
  "visual_events": [
    {"type": "approach", "entities": ["E1", "E2"],
     "start_time": "00:03", "end_time": "00:05",
     "description": "Brown labrador walks toward white poodle",
     "confidence": 0.9}
  ],
  "audio_events": [],
  "multimodal_correlations": [],
  "summary": "Brief description of what happens in the video."
}

Behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe.
Be precise. Include ALL entities and interactions you observe across frames."""


class QwenBackend(BaseVLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 4096,
        max_frames: int = 16,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()
        self._model = None
        self._processor = None
        self._process_vision_info = None

    def _load_model(self):
        """Load model with 4-bit quantization. ~3GB VRAM."""
        if self._model is not None:
            return

        self.logger.info("Loading Qwen2.5-VL-3B with 4-bit quantization...")

        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            from qwen_vl_utils import process_vision_info

            self._process_vision_info = process_vision_info
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install transformers qwen-vl-utils bitsandbytes"
            ) from e

        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load model
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        # Log VRAM usage
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1e9
            vram_reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.info(
                f"Qwen model loaded. VRAM: {vram_allocated:.2f}GB allocated, "
                f"{vram_reserved:.2f}GB reserved"
            )

    def analyze_video(
        self,
        video_path: str,
        frames: Optional[List[Path]] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Process frames through Qwen2.5-VL-3B.

        Steps:
        1. Load model if not loaded
        2. Subsample frames to self.max_frames
        3. Build message with images + prompt
        4. Run inference
        5. Parse JSON response
        6. Return AnalysisResult
        """
        self.logger.info(f"Analyzing video with Qwen2.5-VL-3B")

        # Load model if not loaded
        self._load_model()

        # Use provided prompt or default
        analysis_prompt = prompt or ANALYSIS_PROMPT_LOCAL

        # Subsample frames if needed
        if frames is None:
            raise ValueError("frames must be provided for Qwen backend")

        frame_paths = self._subsample_frames(frames)

        self.logger.info(f"Processing {len(frame_paths)} frames")

        try:
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{frame_path}"}
                        for frame_path in frame_paths
                    ]
                    + [{"type": "text", "text": analysis_prompt}],
                }
            ]

            # Process with processor
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self._process_vision_info(messages)

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            # Generate
            self.logger.info("Running inference...")
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

            # Decode response
            output_ids_trimmed = output_ids[0][inputs.input_ids.shape[1] :]
            response_text = self._processor.decode(
                output_ids_trimmed, skip_special_tokens=True
            )

            self.logger.info("Inference complete")

            # Parse JSON response
            result = self.parser.parse(response_text)

            self.logger.info(
                f"Analysis complete: {len(result.entities)} entities, "
                f"{len(result.visual_events)} visual events"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during Qwen analysis: {e}")
            raise

    def _subsample_frames(self, frames: List[Path]) -> List[Path]:
        """Uniformly subsample frames to max_frames."""
        if len(frames) <= self.max_frames:
            return frames

        indices = [
            int(i * len(frames) / self.max_frames) for i in range(self.max_frames)
        ]
        return [frames[i] for i in indices]

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def cleanup(self):
        """CRITICAL: Free all GPU memory."""
        self.logger.info("Cleaning up Qwen backend...")
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"VRAM after cleanup: {vram_after:.2f}GB")

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def requires_gpu(self) -> bool:
        return True
