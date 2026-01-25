# Forecastly Admin CLI Documentation

The Forecastly CLI provides command-line tools for managing your forecasting system.

## Installation

The CLI is included with Forecastly. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Show help
python cli.py --help

# Show help for a command group
python cli.py db --help

# Run a command
python cli.py db init
```

## Command Groups

- `db` - Database management
- `user` - User management
- `model` - Machine learning model operations
- `data` - Data import/export and cleanup
- `system` - System health and monitoring

---

## Database Commands

### Initialize Database

Create all database tables:

```bash
python cli.py db init
```

### Run Migrations

Apply pending Alembic migrations:

```bash
python cli.py db migrate
```

### Backup Database

Create a SQL backup:

```bash
# Default backup file
python cli.py db backup

# Custom output file
python cli.py db backup --output backups/backup_2024-01-25.sql
```

### Restore Database

Restore from a SQL backup (requires confirmation):

```bash
python cli.py db restore backups/backup_2024-01-25.sql
```

⚠️ **Warning**: This will overwrite current data!

### Reset Database

Drop all tables and reinitialize (requires confirmation):

```bash
python cli.py db reset
```

⚠️ **Danger**: This deletes ALL data permanently!

---

## User Management Commands

### Create User

Create a new user account:

```bash
# Interactive (prompts for password)
python cli.py user create user@example.com

# With options
python cli.py user create admin@example.com \
    --password secretpass \
    --role admin \
    --superuser
```

Options:
- `--password, -p`: User password (prompts if not provided)
- `--role, -r`: User role (admin/analyst/viewer, default: viewer)
- `--superuser`: Make user a superuser

### List Users

Display all users:

```bash
python cli.py user list
```

Output example:
```
📋 Users:
Email                          Role       Active     Superuser  Created
------------------------------------------------------------------------------------------
admin@example.com              admin      ✅         ⭐         2024-01-20
analyst@example.com            analyst    ✅                    2024-01-21
viewer@example.com             viewer     ❌                    2024-01-22
```

### Change User Role

Update a user's role:

```bash
python cli.py user set-role user@example.com --role admin
```

### Reset Password

Reset a user's password:

```bash
python cli.py user reset-password user@example.com
# Prompts for new password
```

### Deactivate User

Deactivate a user account:

```bash
python cli.py user deactivate user@example.com
```

---

## Model Management Commands

### Train Models

Train forecasting models:

```bash
# Train all models
python cli.py model train

# Train specific model
python cli.py model train --type prophet
python cli.py model train --type xgboost
```

### Evaluate Models

Evaluate model performance:

```bash
# Default horizon (14 days)
python cli.py model evaluate

# Custom horizon
python cli.py model evaluate --horizon 30
```

Results are saved to `data/processed/metrics.csv`

### Generate Predictions

Generate forecasts for all SKUs:

```bash
python cli.py model predict
```

Results are saved to `data/processed/predictions.csv`

### List Models

Show available trained models:

```bash
python cli.py model list
```

Output example:
```
📦 Available Models:
  • prophet_model.pkl             15.23 MB    2024-01-25 10:30
    └─ 100 SKU models
  • xgb_model.pkl                 2.45 MB     2024-01-25 10:35
```

---

## Data Management Commands

### Import Sales Data

Import sales from CSV:

```bash
python cli.py data import-sales data/raw/sales.csv
```

Expected CSV format:
```csv
date,sku_id,units,price,promo_flag
2024-01-01,SKU001,100,10.50,false
2024-01-01,SKU002,150,15.00,true
```

### Export Sales Data

Export sales to CSV:

```bash
# Default output
python cli.py data export-sales

# Custom output file
python cli.py data export-sales --output exports/sales_2024.csv
```

### Cleanup Old Data

Remove old predictions and forecast runs:

```bash
# Keep last 90 days (default)
python cli.py data cleanup

# Keep last 30 days
python cli.py data cleanup --days 30
```

⚠️ **Warning**: This permanently deletes old data. Requires confirmation.

---

## System Commands

### Health Check

Check system health:

```bash
python cli.py system health
```

Output example:
```
🏥 Checking system health...

✅ API: Healthy
✅ Database: Connected

📊 System Metrics:
  CPU: 15.3%
  Memory: 42.8%
  Disk: 65.2%
```

### View Metrics

Display current metrics:

```bash
python cli.py system metrics
```

### View Logs

Show recent application logs:

```bash
# Last 50 lines (default)
python cli.py system logs

# Last 100 lines
python cli.py system logs --lines 100
```

Logs are color-coded:
- Red: ERROR
- Yellow: WARNING
- White: INFO
- Gray: DEBUG

---

## Examples and Workflows

### Initial Setup

```bash
# 1. Initialize database
python cli.py db init

# 2. Run migrations
python cli.py db migrate

# 3. Create admin user
python cli.py user create admin@forecastly.com \
    --password secure_password \
    --role admin \
    --superuser

# 4. Import historical sales data
python cli.py data import-sales data/raw/historical_sales.csv

# 5. Train models
python cli.py model train

# 6. Generate initial predictions
python cli.py model predict
```

### Daily Operations

```bash
# Update data and retrain
python cli.py data import-sales data/raw/daily_sales.csv
python cli.py model train --type xgboost
python cli.py model predict

# Check system health
python cli.py system health
```

### Weekly Maintenance

```bash
# Backup database
python cli.py db backup --output backups/weekly_backup_$(date +%Y%m%d).sql

# Clean up old data
python cli.py data cleanup --days 90

# Evaluate model performance
python cli.py model evaluate --horizon 14

# Check logs for errors
python cli.py system logs --lines 200 | grep ERROR
```

### Disaster Recovery

```bash
# 1. Stop application
# 2. Restore from backup
python cli.py db restore backups/last_good_backup.sql

# 3. Verify data
python cli.py user list
python cli.py model list

# 4. Start application
```

---

## Advanced Usage

### Scripting

Use the CLI in shell scripts:

```bash
#!/bin/bash
# daily_update.sh

set -e  # Exit on error

echo "Starting daily update..."

# Import new data
python cli.py data import-sales data/raw/sales_$(date +%Y%m%d).csv

# Retrain models
python cli.py model train

# Generate predictions
python cli.py model predict

# Check health
python cli.py system health

echo "Update complete!"
```

### Cron Jobs

Schedule automated tasks:

```bash
# crontab -e

# Daily backup at 2 AM
0 2 * * * cd /path/to/forecastly && python cli.py db backup --output backups/daily_$(date +\%Y\%m\%d).sql

# Weekly cleanup at 3 AM on Sundays
0 3 * * 0 cd /path/to/forecastly && python cli.py data cleanup --days 90

# Daily model update at 4 AM
0 4 * * * cd /path/to/forecastly && python cli.py model train && python cli.py model predict
```

### Docker Integration

Run CLI commands in Docker:

```bash
# Execute command in running container
docker-compose exec api python cli.py db migrate

# Run one-off command
docker-compose run --rm api python cli.py model train
```

---

## Configuration

### Environment Variables

The CLI uses the same configuration as the main application:

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost/forecastly
SECRET_KEY=your-secret-key
ENVIRONMENT=production
```

### Database Connection

Database URL can be set via:

1. Environment variable: `DATABASE_URL`
2. Configuration file: `src/utils/config.py`
3. Alembic config: `alembic.ini`

---

## Troubleshooting

### Command Not Found

Ensure you're in the project directory:

```bash
cd /path/to/forecastly
python cli.py --help
```

### Database Connection Error

Check database is running and credentials are correct:

```bash
# Test connection
psql -U postgres -d forecastly -c "SELECT 1;"

# Check environment variables
echo $DATABASE_URL
```

### Permission Denied

Make script executable (Unix/Linux):

```bash
chmod +x cli.py
./cli.py --help
```

### Import Errors

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

### Migration Conflicts

If migrations fail:

```bash
# Check current version
alembic current

# Show migration history
alembic history

# Reset migrations (dangerous!)
python cli.py db reset
alembic stamp head
```

---

## Best Practices

1. **Backup Before Major Operations**
   ```bash
   python cli.py db backup --output backups/before_migration.sql
   python cli.py db migrate
   ```

2. **Test on Staging First**
   ```bash
   # On staging
   DATABASE_URL=postgresql://localhost/forecastly_staging python cli.py db migrate
   ```

3. **Use Confirmation Flags Carefully**
   ```bash
   # Safe - requires confirmation
   python cli.py db reset

   # Dangerous - skips confirmation in scripts
   echo "yes" | python cli.py db reset
   ```

4. **Monitor Logs**
   ```bash
   # Check for errors after operations
   python cli.py system logs --lines 100 | grep -E "(ERROR|WARN)"
   ```

5. **Version Control**
   - Never commit `.env` files
   - Commit migration files
   - Document custom CLI commands

---

## Extending the CLI

### Adding Custom Commands

Edit `cli.py` to add new commands:

```python
@cli.command()
@click.argument('parameter')
@click.option('--flag', is_flag=True)
def custom_command(parameter, flag):
    """Your custom command description."""
    click.echo(f"Running custom command: {parameter}")
    # Your implementation
```

### Creating Command Groups

```python
@cli.group()
def mygroup():
    """My custom command group."""
    pass

@mygroup.command()
def subcommand():
    """Subcommand in my group."""
    pass

# Usage: python cli.py mygroup subcommand
```

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/forecastly/issues
- Documentation: `/docs`
- Logs: `python cli.py system logs`
