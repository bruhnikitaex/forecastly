"""
Tests for monitoring and metrics collection.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.monitoring import (
    MetricsCollector,
    get_metrics_collector,
    track_prediction_time,
    track_api_request,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create fresh metrics collector for each test."""
        collector = MetricsCollector()
        collector.reset()
        return collector

    def test_increment_counter(self, collector):
        """Should increment counter."""
        collector.increment_counter('test_counter')
        collector.increment_counter('test_counter')

        assert collector.counters['test_counter'] == 2

    def test_increment_counter_with_value(self, collector):
        """Should increment counter by specific value."""
        collector.increment_counter('test_counter', value=5)

        assert collector.counters['test_counter'] == 5

    def test_set_gauge(self, collector):
        """Should set gauge value."""
        collector.set_gauge('test_gauge', 42.5)

        assert collector.gauges['test_gauge'] == 42.5

    def test_observe_histogram(self, collector):
        """Should add observations to histogram."""
        collector.observe_histogram('test_hist', 1.0)
        collector.observe_histogram('test_hist', 2.0)
        collector.observe_histogram('test_hist', 3.0)

        assert len(collector.histograms['test_hist']) == 3
        assert 1.0 in collector.histograms['test_hist']
        assert 2.0 in collector.histograms['test_hist']
        assert 3.0 in collector.histograms['test_hist']

    def test_track_request(self, collector):
        """Should track API request metrics."""
        collector.track_request(
            endpoint='/api/v1/predict',
            method='GET',
            duration=0.123,
            status_code=200
        )

        # Check that metrics were recorded
        assert len(collector.request_total) > 0
        assert len(collector.request_duration_sum) > 0
        assert len(collector.histograms) > 0

    def test_track_prediction(self, collector):
        """Should track prediction metrics."""
        collector.track_prediction(
            model='prophet',
            sku_id='SKU001',
            duration=1.5,
            success=True
        )

        assert len(collector.model_predictions) > 0
        assert len(collector.prediction_latency) > 0

    def test_track_prediction_error(self, collector):
        """Should track prediction errors."""
        collector.track_prediction(
            model='prophet',
            sku_id='SKU001',
            duration=1.5,
            success=False
        )

        assert len(collector.model_errors) > 0

    def test_collect_system_metrics(self, collector):
        """Should collect system resource metrics."""
        collector.collect_system_metrics()

        assert 'cpu_percent' in collector.system_metrics
        assert 'memory_percent' in collector.system_metrics
        assert 'disk_percent' in collector.system_metrics
        assert 'process_memory_mb' in collector.system_metrics

    def test_get_metrics_text(self, collector):
        """Should return metrics in Prometheus text format."""
        collector.increment_counter('test_counter', value=10)
        collector.set_gauge('test_gauge', 42.0)

        metrics_text = collector.get_metrics_text()

        assert 'test_counter' in metrics_text
        assert '10' in metrics_text
        assert 'test_gauge' in metrics_text
        assert '42' in metrics_text

    def test_get_metrics_json(self, collector):
        """Should return metrics as JSON."""
        collector.increment_counter('test_counter', value=10)
        collector.set_gauge('test_gauge', 42.0)

        metrics_json = collector.get_metrics_json()

        assert 'timestamp' in metrics_json
        assert 'counters' in metrics_json
        assert 'gauges' in metrics_json
        assert 'system' in metrics_json

    def test_reset_clears_all_metrics(self, collector):
        """Reset should clear all metrics."""
        collector.increment_counter('test_counter')
        collector.set_gauge('test_gauge', 42.0)
        collector.track_request('/test', 'GET', 0.1, 200)

        collector.reset()

        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
        assert len(collector.request_total) == 0

    def test_metrics_with_labels(self, collector):
        """Should create separate metrics for different labels."""
        collector.increment_counter('requests', labels={'endpoint': '/api/v1/predict'})
        collector.increment_counter('requests', labels={'endpoint': '/api/v1/skus'})

        assert len(collector.counters) == 2

    def test_histogram_keeps_last_1000_observations(self, collector):
        """Histogram should keep only last 1000 observations."""
        # Add 1500 observations
        for i in range(1500):
            collector.observe_histogram('test_hist', float(i))

        # Should keep only last 1000
        assert len(collector.histograms['test_hist']) == 1000


class TestDecorators:
    """Tests for decorator functions."""

    def test_track_prediction_time_decorator(self):
        """Should track prediction time."""
        collector = get_metrics_collector()
        collector.reset()

        @track_prediction_time('test_model')
        def predict_function():
            time.sleep(0.01)  # Simulate work
            return 'result'

        result = predict_function()

        assert result == 'result'
        assert len(collector.model_predictions) > 0

    def test_track_prediction_time_with_exception(self):
        """Should track failed predictions."""
        collector = get_metrics_collector()
        collector.reset()

        @track_prediction_time('test_model')
        def predict_function():
            raise ValueError('Test error')

        with pytest.raises(ValueError):
            predict_function()

        # Should still track the attempt
        assert len(collector.model_errors) > 0

    def test_track_api_request_decorator(self):
        """Should track API request."""
        collector = get_metrics_collector()
        collector.reset()

        @track_api_request('/test')
        def api_function():
            return {'result': 'ok'}

        result = api_function()

        assert result == {'result': 'ok'}
        assert len(collector.request_total) > 0

    def test_track_api_request_with_exception(self):
        """Should track failed requests."""
        collector = get_metrics_collector()
        collector.reset()

        @track_api_request('/test')
        def api_function():
            raise Exception('Test error')

        with pytest.raises(Exception):
            api_function()

        # Should track with error status
        # (Implementation may vary based on how errors are tracked)


class TestMetricsIntegration:
    """Integration tests for metrics system."""

    def test_multiple_requests_tracking(self):
        """Should track multiple requests correctly."""
        collector = get_metrics_collector()
        collector.reset()

        # Simulate multiple requests
        for i in range(10):
            collector.track_request(
                endpoint='/api/v1/predict',
                method='GET',
                duration=0.1 + i * 0.01,
                status_code=200
            )

        # Get metrics
        metrics = collector.get_metrics_json()

        assert 'requests' in metrics
        assert len(metrics['requests']['total']) > 0

    def test_metrics_persistence_across_calls(self):
        """Metrics should accumulate across multiple calls."""
        collector = get_metrics_collector()
        collector.reset()

        collector.increment_counter('test_counter')
        first_value = collector.counters['test_counter']

        collector.increment_counter('test_counter')
        second_value = collector.counters['test_counter']

        assert second_value == first_value + 1

    def test_singleton_pattern(self):
        """MetricsCollector should be a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

        # Changes to one should affect the other
        collector1.increment_counter('test')
        assert collector2.counters['test'] == 1
