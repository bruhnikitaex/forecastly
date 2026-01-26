"""
Type definitions for the job queue system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


class JobType(str, Enum):
    """Types of background jobs."""
    MODEL_TRAIN_PROPHET = "model_train_prophet"
    MODEL_TRAIN_XGBOOST = "model_train_xgboost"
    MODEL_TRAIN_ALL = "model_train_all"
    PREDICT = "predict"
    PREDICT_REBUILD = "predict_rebuild"
    EVALUATE = "evaluate"
    DATA_IMPORT = "data_import"
    DATA_EXPORT = "data_export"
    REPORT_GENERATE = "report_generate"
    WEBHOOK_DELIVER = "webhook_deliver"
    CLEANUP = "cleanup"


class JobStatus(str, Enum):
    """Status of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority levels for jobs."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class JobResult:
    """Result of a completed job."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class JobProgress:
    """Progress update for a running job."""
    job_id: str
    progress: int  # 0-100
    message: Optional[str] = None
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JobConfig:
    """Configuration for job execution."""
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 3600  # 1 hour
    notify_on_complete: bool = True
    notify_on_failure: bool = True
