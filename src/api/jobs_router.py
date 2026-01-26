"""
API router for background job management.

Provides endpoints for:
- Creating jobs
- Monitoring job status and progress
- Cancelling jobs
- Listing job history
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.utils.logger import logger
from src.db.database import get_db
from src.jobs import get_job_queue, JobType, JobStatus, JobPriority
from src.jobs.worker import get_worker


router = APIRouter(prefix="/jobs", tags=["Jobs"])


# Pydantic models for request/response
class JobCreateRequest(BaseModel):
    """Request to create a new job."""
    job_type: str = Field(..., description="Type of job to create")
    params: dict = Field(default_factory=dict, description="Job parameters")
    priority: str = Field(default="normal", description="Job priority: low, normal, high, critical")

    class Config:
        json_schema_extra = {
            "example": {
                "job_type": "predict",
                "params": {"horizon": 14},
                "priority": "normal"
            }
        }


class JobResponse(BaseModel):
    """Job information response."""
    job_id: str
    job_type: str
    status: str
    priority: int
    progress: int
    progress_message: Optional[str] = None
    params: dict = {}
    result: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for listing jobs."""
    jobs: List[JobResponse]
    total: int
    limit: int
    offset: int


class JobCreateResponse(BaseModel):
    """Response after creating a job."""
    job_id: str
    job_type: str
    status: str
    message: str


# Mapping from string to enum
JOB_TYPE_MAP = {
    "model_train_prophet": JobType.MODEL_TRAIN_PROPHET,
    "model_train_xgboost": JobType.MODEL_TRAIN_XGBOOST,
    "model_train_all": JobType.MODEL_TRAIN_ALL,
    "predict": JobType.PREDICT,
    "predict_rebuild": JobType.PREDICT_REBUILD,
    "evaluate": JobType.EVALUATE,
    "data_import": JobType.DATA_IMPORT,
    "data_export": JobType.DATA_EXPORT,
}

PRIORITY_MAP = {
    "low": JobPriority.LOW,
    "normal": JobPriority.NORMAL,
    "high": JobPriority.HIGH,
    "critical": JobPriority.CRITICAL,
}


def _job_dict_to_response(job: dict) -> JobResponse:
    """Convert job dict to response model."""
    return JobResponse(
        job_id=job["job_id"],
        job_type=job["job_type"],
        status=job["status"],
        priority=job["priority"],
        progress=job.get("progress", 0),
        progress_message=job.get("progress_message"),
        params=job.get("params", {}),
        result=job.get("result"),
        error_message=job.get("error_message"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.post("", response_model=JobCreateResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create a new background job.

    Available job types:
    - model_train_prophet: Train Prophet model
    - model_train_xgboost: Train XGBoost model
    - model_train_all: Train all models
    - predict: Generate predictions
    - predict_rebuild: Rebuild predictions (train + predict)
    - evaluate: Evaluate model accuracy
    - data_export: Export data to file
    """
    # Validate job type
    job_type = JOB_TYPE_MAP.get(request.job_type.lower())
    if not job_type:
        available = list(JOB_TYPE_MAP.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {request.job_type}. Available types: {available}"
        )

    # Validate priority
    priority = PRIORITY_MAP.get(request.priority.lower(), JobPriority.NORMAL)

    # Create the job
    queue = get_job_queue()
    try:
        job_id = queue.create_job(
            job_type=job_type,
            params=request.params,
            priority=priority,
            db=db,
        )

        logger.info(f"Created job {job_id} of type {request.job_type}")

        return JobCreateResponse(
            job_id=job_id,
            job_type=request.job_type,
            status="pending",
            message=f"Job created successfully. Use GET /api/v1/jobs/{job_id} to check status."
        )

    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed, cancelled"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
):
    """
    List background jobs with optional filters.
    """
    queue = get_job_queue()

    # Parse filters
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: pending, running, completed, failed, cancelled"
            )

    type_filter = None
    if job_type:
        type_filter = JOB_TYPE_MAP.get(job_type.lower())
        if not type_filter:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {job_type}"
            )

    try:
        jobs = queue.list_jobs(
            status=status_filter,
            job_type=type_filter,
            limit=limit,
            offset=offset,
            db=db,
        )

        return JobListResponse(
            jobs=[_job_dict_to_response(j) for j in jobs],
            total=len(jobs),  # TODO: Get actual total count
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get job details by ID.
    """
    queue = get_job_queue()

    try:
        job = queue.get_job(job_id, db=db)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return _job_dict_to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Cancel a pending or running job.
    """
    queue = get_job_queue()

    try:
        success = queue.cancel_job(job_id, db=db)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job {job_id}. It may not exist or already completed."
            )

        return {
            "status": "ok",
            "message": f"Job {job_id} cancelled",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/progress")
async def get_job_progress(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get job progress. Useful for polling during long-running jobs.
    """
    queue = get_job_queue()

    try:
        job = queue.get_job(job_id, db=db)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0),
            "progress_message": job.get("progress_message"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job progress {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_jobs(
    days: int = Query(30, ge=1, le=365, description="Delete jobs older than this many days"),
    db: Session = Depends(get_db),
):
    """
    Delete old completed/failed jobs.
    """
    queue = get_job_queue()

    try:
        deleted = queue.cleanup_old_jobs(days=days, db=db)

        return {
            "status": "ok",
            "deleted_count": deleted,
            "message": f"Deleted {deleted} jobs older than {days} days",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error cleaning up jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worker/status")
async def get_worker_status():
    """
    Get background worker status.
    """
    worker = get_worker()

    return {
        "running": worker._running,
        "poll_interval": worker.poll_interval,
        "max_workers": worker.max_workers,
        "registered_handlers": list(worker._handlers.keys()),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/worker/start")
async def start_worker():
    """
    Start the background worker if not running.
    """
    worker = get_worker()

    if worker._running:
        return {
            "status": "already_running",
            "message": "Worker is already running"
        }

    worker.start()

    return {
        "status": "started",
        "message": "Background worker started",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/worker/stop")
async def stop_worker():
    """
    Stop the background worker.
    """
    worker = get_worker()

    if not worker._running:
        return {
            "status": "not_running",
            "message": "Worker is not running"
        }

    worker.stop()

    return {
        "status": "stopped",
        "message": "Background worker stopped",
        "timestamp": datetime.now().isoformat()
    }
