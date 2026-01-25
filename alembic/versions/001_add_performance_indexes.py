"""Add performance indexes

Revision ID: 001
Revises:
Create Date: 2026-01-25

Adds database indexes to improve query performance for common operations:
- Filter by status, dates, and active flags
- Query optimization for frequently accessed columns
- Composite indexes for common join patterns
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes."""

    # ForecastRun indexes
    op.create_index('ix_forecast_runs_status', 'forecast_runs', ['status'], unique=False)
    op.create_index('ix_forecast_runs_started_at', 'forecast_runs', ['started_at'], unique=False)
    op.create_index('ix_forecast_runs_status_started', 'forecast_runs', ['status', 'started_at'], unique=False)

    # SKU indexes
    op.create_index('ix_skus_is_active', 'skus', ['is_active'], unique=False)
    op.create_index('ix_skus_category', 'skus', ['category'], unique=False)
    op.create_index('ix_skus_store_id', 'skus', ['store_id'], unique=False)
    op.create_index('ix_skus_active_category', 'skus', ['is_active', 'category'], unique=False)
    op.create_index('ix_skus_active_store', 'skus', ['is_active', 'store_id'], unique=False)

    # Prediction indexes
    op.create_index('ix_predictions_forecast_run_id', 'predictions', ['forecast_run_id'], unique=False)
    op.create_index('ix_predictions_date_desc', 'predictions', [sa.text('date DESC')], unique=False)
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'], unique=False)

    # Metric indexes
    op.create_index('ix_metrics_forecast_run_id', 'metrics', ['forecast_run_id'], unique=False)
    op.create_index('ix_metrics_best_model', 'metrics', ['best_model'], unique=False)
    op.create_index('ix_metrics_created_at', 'metrics', ['created_at'], unique=False)
    op.create_index('ix_metrics_sku_created', 'metrics', ['sku_id', 'created_at'], unique=False)

    # SalesHistory indexes
    op.create_index('ix_sales_date_desc', 'sales_history', [sa.text('date DESC')], unique=False)
    op.create_index('ix_sales_promo_flag', 'sales_history', ['promo_flag'], unique=False)
    op.create_index('ix_sales_created_at', 'sales_history', ['created_at'], unique=False)

    # User indexes
    op.create_index('ix_users_role', 'users', ['role'], unique=False)
    op.create_index('ix_users_is_active', 'users', ['is_active'], unique=False)
    op.create_index('ix_users_created_at', 'users', ['created_at'], unique=False)
    op.create_index('ix_users_active_role', 'users', ['is_active', 'role'], unique=False)
    op.create_index('ix_users_last_login', 'users', ['last_login'], unique=False)
    op.create_index('ix_users_locked_until', 'users', ['locked_until'], unique=False)

    # APIKey indexes
    op.create_index('ix_api_keys_is_active', 'api_keys', ['is_active'], unique=False)
    op.create_index('ix_api_keys_expires_at', 'api_keys', ['expires_at'], unique=False)
    op.create_index('ix_api_keys_user_active', 'api_keys', ['user_id', 'is_active'], unique=False)
    op.create_index('ix_api_keys_last_used', 'api_keys', ['last_used_at'], unique=False)

    # RefreshToken indexes
    op.create_index('ix_refresh_tokens_is_revoked', 'refresh_tokens', ['is_revoked'], unique=False)
    op.create_index('ix_refresh_tokens_expires_at', 'refresh_tokens', ['expires_at'], unique=False)
    op.create_index('ix_refresh_tokens_user_revoked', 'refresh_tokens', ['user_id', 'is_revoked'], unique=False)


def downgrade() -> None:
    """Remove performance indexes."""

    # RefreshToken indexes
    op.drop_index('ix_refresh_tokens_user_revoked', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_expires_at', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_is_revoked', table_name='refresh_tokens')

    # APIKey indexes
    op.drop_index('ix_api_keys_last_used', table_name='api_keys')
    op.drop_index('ix_api_keys_user_active', table_name='api_keys')
    op.drop_index('ix_api_keys_expires_at', table_name='api_keys')
    op.drop_index('ix_api_keys_is_active', table_name='api_keys')

    # User indexes
    op.drop_index('ix_users_locked_until', table_name='users')
    op.drop_index('ix_users_last_login', table_name='users')
    op.drop_index('ix_users_active_role', table_name='users')
    op.drop_index('ix_users_created_at', table_name='users')
    op.drop_index('ix_users_is_active', table_name='users')
    op.drop_index('ix_users_role', table_name='users')

    # SalesHistory indexes
    op.drop_index('ix_sales_created_at', table_name='sales_history')
    op.drop_index('ix_sales_promo_flag', table_name='sales_history')
    op.drop_index('ix_sales_date_desc', table_name='sales_history')

    # Metric indexes
    op.drop_index('ix_metrics_sku_created', table_name='metrics')
    op.drop_index('ix_metrics_created_at', table_name='metrics')
    op.drop_index('ix_metrics_best_model', table_name='metrics')
    op.drop_index('ix_metrics_forecast_run_id', table_name='metrics')

    # Prediction indexes
    op.drop_index('ix_predictions_created_at', table_name='predictions')
    op.drop_index('ix_predictions_date_desc', table_name='predictions')
    op.drop_index('ix_predictions_forecast_run_id', table_name='predictions')

    # SKU indexes
    op.drop_index('ix_skus_active_store', table_name='skus')
    op.drop_index('ix_skus_active_category', table_name='skus')
    op.drop_index('ix_skus_store_id', table_name='skus')
    op.drop_index('ix_skus_category', table_name='skus')
    op.drop_index('ix_skus_is_active', table_name='skus')

    # ForecastRun indexes
    op.drop_index('ix_forecast_runs_status_started', table_name='forecast_runs')
    op.drop_index('ix_forecast_runs_started_at', table_name='forecast_runs')
    op.drop_index('ix_forecast_runs_status', table_name='forecast_runs')
