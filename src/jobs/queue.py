"""
Job queue management for background tasks.
"""

import uuid
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import asdict

from sqlalchemy.orm import Session

from src.utils.logger import logger
from src.db.database import get_db_session
from src.db.models import BackgroundJob
from .types import JobType, JobStatus, JobPriority, JobResult, JobProgress


class JobQueue:
    """
    Manages background job queue.

    Supports creating, updating, and querying jobs from the database.
    """

    def create_job(
        self,
        job_type: JobType,
        params: Dict[str, Any],
        tenant_id: Optional[int] = None,
        user_id: Optional[int] = None,
        priority: JobPriority = JobPriority.NORMAL,
        db: Optional[Session] = None,
    ) -> str:
        """
        Create a new background job.

        Args:
            job_type: Type of job to create
            params: Parameters for the job
            tenant_id: Optional tenant ID
            user_id: Optional user ID who created the job
            priority: Job priority
            db: Optional database session

        Returns:
            Job ID (UUID string)
        """
        job_id = str(uuid.uuid4())

        def _create(session: Session):
            job = BackgroundJob(
                job_id=job_id,
                tenant_id=tenant_id,
                user_id=user_id,
                job_type=job_type.value,
                status=JobStatus.PENDING.value,
                priority=priority.value,
                params=json.dumps(params),
            )
            session.add(job)
            session.commit()
            logger.info(f"Created job {job_id} of type {job_type.value}")
            return job_id

        if db:
            return _create(db)
        else:
            with get_db_session() as session:
                return _create(session)

    def get_job(self, job_id: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Get job details by ID.

        Args:
            job_id: Job ID
            db: Optional database session

        Returns:
            Job details dict or None if not found
        """
        def _get(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.job_id == job_id
            ).first()

            if not job:
                return None

            return self._job_to_dict(job)

        if db:
            return _get(db)
        else:
            with get_db_session() as session:
                return _get(session)

    def list_jobs(
        self,
        tenant_id: Optional[int] = None,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
        offset: int = 0,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filters.

        Args:
            tenant_id: Filter by tenant
            status: Filter by status
            job_type: Filter by job type
            limit: Max results
            offset: Pagination offset
            db: Optional database session

        Returns:
            List of job dicts
        """
        def _list(session: Session):
            query = session.query(BackgroundJob)

            if tenant_id is not None:
                query = query.filter(BackgroundJob.tenant_id == tenant_id)
            if status is not None:
                query = query.filter(BackgroundJob.status == status.value)
            if job_type is not None:
                query = query.filter(BackgroundJob.job_type == job_type.value)

            query = query.order_by(
                BackgroundJob.priority.desc(),
                BackgroundJob.created_at.asc()
            )

            jobs = query.offset(offset).limit(limit).all()
            return [self._job_to_dict(job) for job in jobs]

        if db:
            return _list(db)
        else:
            with get_db_session() as session:
                return _list(session)

    def get_next_pending_job(self, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next pending job to process (highest priority, oldest first).

        Returns:
            Job dict or None if no pending jobs
        """
        def _get_next(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.status == JobStatus.PENDING.value
            ).order_by(
                BackgroundJob.priority.desc(),
                BackgroundJob.created_at.asc()
            ).first()

            if not job:
                return None

            return self._job_to_dict(job)

        if db:
            return _get_next(db)
        else:
            with get_db_session() as session:
                return _get_next(session)

    def start_job(self, job_id: str, db: Optional[Session] = None) -> bool:
        """
        Mark a job as started.

        Returns:
            True if successful
        """
        def _start(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.job_id == job_id
            ).first()

            if not job:
                return False

            job.status = JobStatus.RUNNING.value
            job.started_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(f"Started job {job_id}")
            return True

        if db:
            return _start(db)
        else:
            with get_db_session() as session:
                return _start(session)

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> bool:
        """
        Update job progress.

        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
            message: Optional progress message

        Returns:
            True if successful
        """
        def _update(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.job_id == job_id
            ).first()

            if not job:
                return False

            job.progress = min(100, max(0, progress))
            if message:
                job.progress_message = message
            session.commit()
            return True

        if db:
            return _update(db)
        else:
            with get_db_session() as session:
                return _update(session)

    def complete_job(
        self,
        job_id: str,
        result: JobResult,
        db: Optional[Session] = None,
    ) -> bool:
        """
        Mark a job as completed.

        Args:
            job_id: Job ID
            result: Job result

        Returns:
            True if successful
        """
        def _complete(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.job_id == job_id
            ).first()

            if not job:
                return False

            job.status = JobStatus.COMPLETED.value if result.success else JobStatus.FAILED.value
            job.completed_at = datetime.now(timezone.utc)
            job.progress = 100 if result.success else job.progress
            job.result = json.dumps({
                "success": result.success,
                "data": result.data,
                "warnings": result.warnings,
                "duration_seconds": result.duration_seconds,
            })
            if result.error:
                job.error_message = result.error

            session.commit()
            logger.info(f"Completed job {job_id} with status {'success' if result.success else 'failed'}")
            return True

        if db:
            return _complete(db)
        else:
            with get_db_session() as session:
                return _complete(session)

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        db: Optional[Session] = None,
    ) -> bool:
        """
        Mark a job as failed.

        Args:
            job_id: Job ID
            error_message: Error description

        Returns:
            True if successful
        """
        result = JobResult(success=False, error=error_message)
        return self.complete_job(job_id, result, db)

    def cancel_job(self, job_id: str, db: Optional[Session] = None) -> bool:
        """
        Cancel a pending or running job.

        Returns:
            True if successful
        """
        def _cancel(session: Session):
            job = session.query(BackgroundJob).filter(
                BackgroundJob.job_id == job_id,
                BackgroundJob.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value])
            ).first()

            if not job:
                return False

            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(f"Cancelled job {job_id}")
            return True

        if db:
            return _cancel(db)
        else:
            with get_db_session() as session:
                return _cancel(session)

    def cleanup_old_jobs(self, days: int = 30, db: Optional[Session] = None) -> int:
        """
        Delete completed/failed jobs older than specified days.

        Returns:
            Number of deleted jobs
        """
        from datetime import timedelta

        def _cleanup(session: Session):
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            result = session.query(BackgroundJob).filter(
                BackgroundJob.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]),
                BackgroundJob.completed_at < cutoff
            ).delete()
            session.commit()
            logger.info(f"Cleaned up {result} old jobs")
            return result

        if db:
            return _cleanup(db)
        else:
            with get_db_session() as session:
                return _cleanup(session)

    def _job_to_dict(self, job: BackgroundJob) -> Dict[str, Any]:
        """Convert job model to dict."""
        return {
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "user_id": job.user_id,
            "job_type": job.job_type,
            "status": job.status,
            "priority": job.priority,
            "params": json.loads(job.params) if job.params else {},
            "result": json.loads(job.result) if job.result else None,
            "progress": job.progress,
            "progress_message": job.progress_message,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
