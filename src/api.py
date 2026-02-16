from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
from src.models import PipelineConfig, PipelineResult
import torch

app = FastAPI(
    title="TempoGraph API", description="Video Intelligence Pipeline", version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (for hackathon scope)
jobs: dict = {}  # job_id -> {"status": "...", "result": ..., "error": ...}


@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    backend: str = Form("gemini"),
    modules: str = Form("behavior,detection,audio"),
    fps: float = Form(1.0),
    max_frames: int = Form(60),
    confidence: float = Form(0.5),
):
    """
    Upload video and start analysis.
    Returns job_id for status polling.
    """
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded video to temp location
    video_path = f"/tmp/tempograph_{job_id}/{video.filename}"
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Parse modules string to dict
    module_list = [m.strip() for m in modules.split(",")]
    module_dict = {
        "behavior": "behavior" in module_list,
        "detection": "detection" in module_list,
        "depth": "depth" in module_list,
        "audio": "audio" in module_list,
    }

    config = PipelineConfig(
        backend=backend,
        modules=module_dict,
        fps=fps,
        max_frames=max_frames,
        confidence=confidence,
        video_path=video_path,
        output_dir=f"results/{job_id}",
    )

    jobs[job_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(run_pipeline_job, job_id, config)

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return {"job_id": job_id, "status": jobs[job_id]["status"]}


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    if job["status"] == "processing":
        return {"status": "processing"}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "complete", "result": job["result"]}


async def run_pipeline_job(job_id: str, config: PipelineConfig):
    """Background task that runs the pipeline."""
    try:
        # Import here to avoid circular imports
        from src.pipeline import Pipeline

        pipeline = Pipeline(config)
        result = pipeline.run()
        jobs[job_id] = {
            "status": "complete",
            "result": result.model_dump(),
            "error": None,
        }
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
