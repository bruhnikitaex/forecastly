"""
Job worker for processing background tasks.
"""

import time
import threading
import traceback
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from src.utils.logger import logger
from .queue import get_job_queue, JobQueue
from .types import JobType, JobStatus, JobResult


class JobWorker:
    """
    Worker that processes background jobs from the queue.

    Runs in a separate thread and continuously polls for new jobs.
    """

    def __init__(
        self,
        queue: Optional[JobQueue] = None,
        poll_interval: float = 5.0,
        max_workers: int = 1,
    ):
        """
        Initialize the job worker.

        Args:
            queue: Job queue instance (uses default if not provided)
            poll_interval: Seconds between queue polls
            max_workers: Max concurrent jobs (currently 1)
        """
        self.queue = queue or get_job_queue()
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register handlers for built-in job types."""
        self.register_handler(JobType.MODEL_TRAIN_PROPHET, self._handle_train_prophet)
        self.register_handler(JobType.MODEL_TRAIN_XGBOOST, self._handle_train_xgboost)
        self.register_handler(JobType.MODEL_TRAIN_ALL, self._handle_train_all)
        self.register_handler(JobType.PREDICT, self._handle_predict)
        self.register_handler(JobType.PREDICT_REBUILD, self._handle_predict_rebuild)
        self.register_handler(JobType.EVALUATE, self._handle_evaluate)
        self.register_handler(JobType.DATA_EXPORT, self._handle_export)

    def register_handler(self, job_type: JobType, handler: Callable):
        """
        Register a handler function for a job type.

        Args:
            job_type: Type of job
            handler: Function that takes (job_id, params) and returns JobResult
        """
        self._handlers[job_type.value] = handler
        logger.debug(f"Registered handler for {job_type.value}")

    def start(self):
        """Start the worker thread."""
        if self._running:
            logger.warning("Worker already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Job worker started")

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Job worker stopped")

    def _run_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                job = self.queue.get_next_pending_job()

                if job:
                    self._process_job(job)
                else:
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(self.poll_interval)

    def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_id = job["job_id"]
        job_type = job["job_type"]
        params = job.get("params", {})

        logger.info(f"Processing job {job_id} of type {job_type}")

        # Mark as running
        self.queue.start_job(job_id)

        start_time = datetime.utcnow()

        try:
            handler = self._handlers.get(job_type)

            if not handler:
                raise ValueError(f"No handler for job type: {job_type}")

            # Execute the handler
            result = handler(job_id, params)

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration

            # Complete the job
            self.queue.complete_job(job_id, result)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            self.queue.fail_job(job_id, error_msg)

    # Default handlers

    def _handle_train_prophet(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle Prophet model training."""
        try:
            from src.models.train_prophet import train

            self.queue.update_progress(job_id, 10, "Loading data...")
            self.queue.update_progress(job_id, 30, "Training Prophet models...")

            train()

            self.queue.update_progress(job_id, 100, "Training complete")
            return JobResult(success=True, data={"model": "prophet"})

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_train_xgboost(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle XGBoost model training."""
        try:
            from src.models.train_xgboost import train

            self.queue.update_progress(job_id, 10, "Loading data...")
            self.queue.update_progress(job_id, 30, "Training XGBoost model...")

            train()

            self.queue.update_progress(job_id, 100, "Training complete")
            return JobResult(success=True, data={"model": "xgboost"})

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_train_all(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle training all models."""
        try:
            from src.models.train_prophet import train as train_prophet
            from src.models.train_xgboost import train as train_xgboost

            self.queue.update_progress(job_id, 10, "Training Prophet...")
            train_prophet()

            self.queue.update_progress(job_id, 50, "Training XGBoost...")
            train_xgboost()

            self.queue.update_progress(job_id, 100, "All models trained")
            return JobResult(success=True, data={"models": ["prophet", "xgboost"]})

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_predict(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle prediction generation."""
        try:
            from src.models.predict import predict

            horizon = params.get("horizon", 14)

            self.queue.update_progress(job_id, 10, "Loading models...")
            self.queue.update_progress(job_id, 30, f"Generating predictions (horizon={horizon})...")

            output_path = predict(horizon=horizon)

            self.queue.update_progress(job_id, 100, "Predictions complete")
            return JobResult(
                success=True,
                data={"output_path": str(output_path), "horizon": horizon}
            )

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_predict_rebuild(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle full prediction rebuild."""
        try:
            from src.models.train_prophet import train as train_prophet
            from src.models.train_xgboost import train as train_xgboost
            from src.models.predict import predict

            horizon = params.get("horizon", 14)

            self.queue.update_progress(job_id, 10, "Training Prophet...")
            train_prophet()

            self.queue.update_progress(job_id, 40, "Training XGBoost...")
            train_xgboost()

            self.queue.update_progress(job_id, 70, "Generating predictions...")
            output_path = predict(horizon=horizon)

            self.queue.update_progress(job_id, 100, "Rebuild complete")
            return JobResult(
                success=True,
                data={"output_path": str(output_path), "horizon": horizon}
            )

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_evaluate(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle model evaluation."""
        try:
            from src.models.evaluate import evaluate

            horizon = params.get("horizon", 14)

            self.queue.update_progress(job_id, 10, "Loading data...")
            self.queue.update_progress(job_id, 30, "Evaluating models...")

            evaluate(horizon=horizon)

            self.queue.update_progress(job_id, 100, "Evaluation complete")
            return JobResult(success=True, data={"horizon": horizon})

        except Exception as e:
            return JobResult(success=False, error=str(e))

    def _handle_export(self, job_id: str, params: Dict[str, Any]) -> JobResult:
        """Handle data export."""
        try:
            import pandas as pd
            from pathlib import Path

            export_type = params.get("type", "predictions")
            format = params.get("format", "csv")

            self.queue.update_progress(job_id, 20, f"Exporting {export_type}...")

            # Load data based on type
            if export_type == "predictions":
                data_path = Path("data/processed/predictions.csv")
            elif export_type == "metrics":
                data_path = Path("data/processed/metrics.csv")
            else:
                raise ValueError(f"Unknown export type: {export_type}")

            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)

            # Export to requested format
            output_dir = Path("data/exports")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{export_type}_{timestamp}.{format}"

            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", date_format="iso")
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unknown export format: {format}")

            self.queue.update_progress(job_id, 100, "Export complete")
            return JobResult(
                success=True,
                data={"output_path": str(output_path), "rows": len(df)}
            )

        except Exception as e:
            return JobResult(success=False, error=str(e))


# Global worker instance
_worker: Optional[JobWorker] = None


def get_worker() -> JobWorker:
    """Get the global worker instance."""
    global _worker
    if _worker is None:
        _worker = JobWorker()
    return _worker


def start_worker():
    """Start the global worker."""
    worker = get_worker()
    worker.start()
    return worker
