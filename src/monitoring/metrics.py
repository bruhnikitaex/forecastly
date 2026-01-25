"""
Metrics collection for monitoring application performance.

Provides Prometheus-compatible metrics for:
- API request latency
- Model prediction time
- Data quality metrics
- System resource usage
"""

import time
from typing import Dict, Optional, Callable
from functools import wraps
from collections import defaultdict
from datetime import datetime
import psutil
import os

from src.utils.logger import logger


class MetricsCollector:
    """
    Collects application metrics in Prometheus format.

    Metrics include:
    - Counters: Total number of events
    - Gauges: Current value of a metric
    - Histograms: Distribution of values
    - Summaries: Statistical summaries
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Counters
        self.counters = defaultdict(int)

        # Gauges (current values)
        self.gauges = defaultdict(float)

        # Histograms (buckets for latency tracking)
        self.histograms = defaultdict(list)

        # Request tracking
        self.request_total = defaultdict(int)
        self.request_duration_sum = defaultdict(float)
        self.request_duration_count = defaultdict(int)

        # Model performance
        self.model_predictions = defaultdict(int)
        self.model_errors = defaultdict(int)
        self.prediction_latency = defaultdict(list)

        # System metrics
        self.system_metrics = {}

        logger.info("Metrics collector initialized")

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge metric to a specific value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add an observation to a histogram."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        # Keep only last 1000 observations per histogram
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

    def track_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """Track API request metrics."""
        labels = {'endpoint': endpoint, 'method': method, 'status': str(status_code)}
        key = self._make_key('http_requests', labels)

        self.request_total[key] += 1
        self.request_duration_sum[key] += duration
        self.request_duration_count[key] += 1

        # Also track in histogram
        self.observe_histogram('http_request_duration_seconds', duration, labels)

    def track_prediction(self, model: str, sku_id: str, duration: float, success: bool):
        """Track model prediction metrics."""
        labels = {'model': model, 'sku_id': sku_id}
        key = self._make_key('predictions', labels)

        self.model_predictions[key] += 1

        if not success:
            self.model_errors[key] += 1

        self.prediction_latency[key].append(duration)

        # Keep only last 100 latency measurements per SKU
        if len(self.prediction_latency[key]) > 100:
            self.prediction_latency[key] = self.prediction_latency[key][-100:]

    def collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            self.system_metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_percent'] = memory.percent
            self.system_metrics['memory_used_mb'] = memory.used / (1024 * 1024)

            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics['disk_percent'] = disk.percent

            # Process info
            process = psutil.Process(os.getpid())
            self.system_metrics['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            self.system_metrics['process_cpu_percent'] = process.cpu_percent(interval=0.1)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        lines = []

        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value:.4f}")

        # Request metrics
        for key, count in self.request_total.items():
            lines.append(f"{key}_total {count}")

            if count > 0:
                avg_duration = self.request_duration_sum[key] / count
                lines.append(f"{key}_duration_seconds_avg {avg_duration:.4f}")

        # Histogram summaries
        for name, values in self.histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values):.4f}")

                # Quantiles
                import numpy as np
                sorted_values = sorted(values)
                p50 = np.percentile(sorted_values, 50)
                p95 = np.percentile(sorted_values, 95)
                p99 = np.percentile(sorted_values, 99)

                lines.append(f"{name}_p50 {p50:.4f}")
                lines.append(f"{name}_p95 {p95:.4f}")
                lines.append(f"{name}_p99 {p99:.4f}")

        # System metrics
        self.collect_system_metrics()
        for name, value in self.system_metrics.items():
            lines.append(f"# TYPE system_{name} gauge")
            lines.append(f"system_{name} {value:.4f}")

        return "\n".join(lines)

    def get_metrics_json(self) -> Dict:
        """
        Get metrics as JSON dictionary.

        Returns:
            Dictionary with all metrics
        """
        self.collect_system_metrics()

        return {
            'timestamp': datetime.now().isoformat(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'requests': {
                'total': dict(self.request_total),
                'duration_avg': {
                    k: v / self.request_duration_count[k]
                    for k, v in self.request_duration_sum.items()
                    if self.request_duration_count[k] > 0
                }
            },
            'predictions': {
                'total': dict(self.model_predictions),
                'errors': dict(self.model_errors),
            },
            'system': self.system_metrics,
        }

    def _make_key(self, name: str, labels: Optional[Dict] = None) -> str:
        """Create a metric key with labels."""
        if labels is None:
            return name

        label_str = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.request_total.clear()
        self.request_duration_sum.clear()
        self.request_duration_count.clear()
        self.model_predictions.clear()
        self.model_errors.clear()
        self.prediction_latency.clear()
        self.system_metrics.clear()


# Global instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


# Decorators

def track_prediction_time(model_name: str):
    """
    Decorator to track model prediction time.

    Args:
        model_name: Name of the model
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time

                # Try to extract SKU ID from args/kwargs
                sku_id = 'unknown'
                if args and hasattr(args[0], '__name__'):
                    sku_id = args[0].__name__
                elif 'sku_id' in kwargs:
                    sku_id = kwargs['sku_id']

                _metrics_collector.track_prediction(
                    model=model_name,
                    sku_id=sku_id,
                    duration=duration,
                    success=success
                )

        return wrapper
    return decorator


def track_api_request(endpoint: str):
    """
    Decorator to track API request metrics.

    Args:
        endpoint: API endpoint name
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time

                _metrics_collector.track_request(
                    endpoint=endpoint,
                    method='GET',  # Could be extracted from request
                    duration=duration,
                    status_code=status_code
                )

        return wrapper
    return decorator


def track_model_performance(model_name: str, sku_id: str, mape: float):
    """
    Track model performance metric.

    Args:
        model_name: Name of the model
        sku_id: SKU identifier
        mape: MAPE value
    """
    labels = {'model': model_name, 'sku_id': sku_id}
    _metrics_collector.set_gauge('model_mape', mape, labels)
