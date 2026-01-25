# Database Indexes Documentation

This document describes the database indexes used in Forecastly to optimize query performance.

## Overview

Proper indexing is crucial for database performance, especially as data grows. This document explains which indexes are created, why they're needed, and how to maintain them.

## Index Strategy

Indexes are created based on:
1. **Common query patterns** - Frequently used WHERE clauses
2. **Foreign key relationships** - JOIN operations
3. **Sorting requirements** - ORDER BY clauses
4. **Uniqueness constraints** - Ensure data integrity

## Indexes by Table

### SKU Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_skus_sku_id` | sku_id | Unique | Primary lookup by SKU identifier |
| `ix_skus_is_active` | is_active | Non-unique | Filter active/inactive products |
| `ix_skus_category` | category | Non-unique | Group by category queries |
| `ix_skus_store_id` | store_id | Non-unique | Filter by store |
| `ix_skus_active_category` | is_active, category | Composite | Filter active products by category |
| `ix_skus_active_store` | is_active, store_id | Composite | Filter active products by store |

**Common Queries Optimized:**
```sql
-- Get active SKUs
SELECT * FROM skus WHERE is_active = true;

-- Get SKUs by category
SELECT * FROM skus WHERE category = 'Electronics' AND is_active = true;

-- Get store-specific SKUs
SELECT * FROM skus WHERE store_id = 'STORE001' AND is_active = true;
```

### Prediction Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_predictions_sku_id` | sku_id | Foreign Key | JOIN with SKU table |
| `ix_predictions_date` | date | Non-unique | Filter by date range |
| `ix_predictions_sku_date` | sku_id, date | Composite | Unique constraint + lookup |
| `ix_predictions_forecast_run_id` | forecast_run_id | Foreign Key | Filter by forecast run |
| `ix_predictions_date_desc` | date DESC | Descending | Get latest predictions |
| `ix_predictions_created_at` | created_at | Non-unique | Audit queries |

**Common Queries Optimized:**
```sql
-- Get predictions for a SKU
SELECT * FROM predictions WHERE sku_id = 123 ORDER BY date DESC;

-- Get latest predictions
SELECT * FROM predictions WHERE date >= '2024-01-01' ORDER BY date DESC;

-- Get predictions from a specific run
SELECT * FROM predictions WHERE forecast_run_id = 456;
```

### ForecastRun Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_forecast_runs_run_id` | run_id | Unique | Primary lookup |
| `ix_forecast_runs_status` | status | Non-unique | Filter by status |
| `ix_forecast_runs_started_at` | started_at | Non-unique | Time-based queries |
| `ix_forecast_runs_status_started` | status, started_at | Composite | Recent runs by status |

**Common Queries Optimized:**
```sql
-- Get running forecasts
SELECT * FROM forecast_runs WHERE status = 'running';

-- Get recent completed runs
SELECT * FROM forecast_runs
WHERE status = 'completed'
ORDER BY started_at DESC
LIMIT 10;
```

### Metric Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_metrics_sku_id` | sku_id | Foreign Key | JOIN with SKU table |
| `ix_metrics_forecast_run_id` | forecast_run_id | Foreign Key | Filter by run |
| `ix_metrics_best_model` | best_model | Non-unique | Analyze model performance |
| `ix_metrics_created_at` | created_at | Non-unique | Time-based queries |
| `ix_metrics_sku_created` | sku_id, created_at | Composite | Historical metrics |

**Common Queries Optimized:**
```sql
-- Get metrics for a SKU
SELECT * FROM metrics WHERE sku_id = 123 ORDER BY created_at DESC;

-- Find all SKUs where Prophet performs best
SELECT * FROM metrics WHERE best_model = 'prophet';
```

### SalesHistory Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_sales_sku_id` | sku_id | Foreign Key | JOIN with SKU table |
| `ix_sales_date` | date | Non-unique | Date range queries |
| `ix_sales_sku_date` | sku_id, date | Composite | Unique constraint + lookup |
| `ix_sales_date_desc` | date DESC | Descending | Recent sales |
| `ix_sales_promo_flag` | promo_flag | Non-unique | Analyze promotions |
| `ix_sales_created_at` | created_at | Non-unique | Audit queries |

**Common Queries Optimized:**
```sql
-- Get sales history for a SKU
SELECT * FROM sales_history
WHERE sku_id = 123 AND date >= '2024-01-01'
ORDER BY date DESC;

-- Analyze promotional periods
SELECT * FROM sales_history WHERE promo_flag = true;
```

### User Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_users_email` | email | Unique | Login lookup |
| `ix_users_role` | role | Non-unique | Filter by role |
| `ix_users_is_active` | is_active | Non-unique | Filter active users |
| `ix_users_created_at` | created_at | Non-unique | Registration time queries |
| `ix_users_active_role` | is_active, role | Composite | Active users by role |
| `ix_users_last_login` | last_login | Non-unique | Inactive user cleanup |
| `ix_users_locked_until` | locked_until | Non-unique | Account lockout queries |

**Common Queries Optimized:**
```sql
-- Get active admins
SELECT * FROM users WHERE is_active = true AND role = 'admin';

-- Find locked accounts
SELECT * FROM users WHERE locked_until > NOW();
```

### APIKey Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_api_keys_key_hash` | key_hash | Unique | API key lookup |
| `ix_api_keys_user_id` | user_id | Foreign Key | User's keys |
| `ix_api_keys_is_active` | is_active | Non-unique | Filter active keys |
| `ix_api_keys_expires_at` | expires_at | Non-unique | Expiration checks |
| `ix_api_keys_user_active` | user_id, is_active | Composite | User's active keys |
| `ix_api_keys_last_used` | last_used_at | Non-unique | Usage analytics |

**Common Queries Optimized:**
```sql
-- Get user's active keys
SELECT * FROM api_keys WHERE user_id = 123 AND is_active = true;

-- Find expired keys
SELECT * FROM api_keys WHERE expires_at < NOW();
```

### RefreshToken Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_refresh_tokens_token_hash` | token_hash | Unique | Token lookup |
| `ix_refresh_tokens_user_id` | user_id | Foreign Key | User's tokens |
| `ix_refresh_tokens_is_revoked` | is_revoked | Non-unique | Filter revoked tokens |
| `ix_refresh_tokens_expires_at` | expires_at | Non-unique | Expiration checks |
| `ix_refresh_tokens_user_revoked` | user_id, is_revoked | Composite | Valid user tokens |

**Common Queries Optimized:**
```sql
-- Get user's valid tokens
SELECT * FROM refresh_tokens
WHERE user_id = 123
  AND is_revoked = false
  AND expires_at > NOW();
```

### SecurityAuditLog Table

| Index Name | Columns | Type | Purpose |
|------------|---------|------|---------|
| `ix_audit_event_type` | event_type | Non-unique | Filter by event |
| `ix_audit_user_id` | user_id | Foreign Key | User activity |
| `ix_audit_user_email` | user_email | Non-unique | Email-based search |
| `ix_audit_created_at` | created_at | Non-unique | Time-based queries |
| `ix_audit_event_created` | event_type, created_at | Composite | Recent events by type |
| `ix_audit_user_created` | user_id, created_at | Composite | User history |
| `ix_audit_severity_created` | severity, created_at | Composite | Critical events |

**Common Queries Optimized:**
```sql
-- Get recent failed logins
SELECT * FROM security_audit_logs
WHERE event_type = 'login_failed'
ORDER BY created_at DESC
LIMIT 100;

-- Get critical events
SELECT * FROM security_audit_logs
WHERE severity = 'critical'
ORDER BY created_at DESC;
```

## Performance Considerations

### Index Size

Indexes consume disk space and memory. Monitor index sizes:

```sql
-- PostgreSQL: Check index sizes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Index Maintenance

PostgreSQL automatically maintains indexes, but periodic maintenance helps:

```sql
-- Rebuild indexes (during maintenance window)
REINDEX TABLE predictions;

-- Update statistics
ANALYZE predictions;

-- Full vacuum (reclaims space)
VACUUM FULL predictions;
```

### Monitoring Index Usage

Track which indexes are actually used:

```sql
-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

## When to Add New Indexes

Consider adding an index when:
1. Query execution time is slow (> 100ms)
2. Table scan appears in EXPLAIN output
3. WHERE clause filters on unindexed column
4. JOIN performance is poor
5. ORDER BY causes filesort

**Don't index:**
- Very small tables (< 1000 rows)
- Columns with low cardinality (few distinct values)
- Frequently updated columns (slows writes)
- Rarely queried columns

## Migration Management

All index changes are managed via Alembic migrations.

### Running Migrations

```bash
# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1

# Show migration history
alembic history

# Show current version
alembic current
```

### Creating New Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new index"

# Create empty migration
alembic revision -m "Custom migration"
```

## Best Practices

1. **Test Before Production**: Always test migrations on a copy of production data
2. **Index Names**: Use descriptive names: `ix_tablename_column`
3. **Composite Indexes**: Order matters - most selective column first
4. **Monitor Performance**: Track query performance before and after
5. **Regular Reviews**: Periodically review and remove unused indexes
6. **Document Changes**: Update this document when adding/removing indexes

## Troubleshooting

### Slow Queries After Migration

```sql
-- Check if index is being used
EXPLAIN ANALYZE
SELECT * FROM predictions WHERE sku_id = 123;
```

### Missing Index

```sql
-- Manually create index if needed
CREATE INDEX CONCURRENTLY ix_predictions_custom
ON predictions(sku_id, date)
WHERE is_active = true;
```

### Bloated Indexes

```sql
-- Find bloated indexes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE pg_relation_size(indexrelid) > 100000000  -- > 100MB
ORDER BY pg_relation_size(indexrelid) DESC;

-- Rebuild if needed
REINDEX INDEX CONCURRENTLY ix_predictions_sku_date;
```

## References

- [PostgreSQL Indexes Documentation](https://www.postgresql.org/docs/current/indexes.html)
- [Alembic Migration Guide](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [SQLAlchemy Indexing](https://docs.sqlalchemy.org/en/14/core/constraints.html#indexes)
