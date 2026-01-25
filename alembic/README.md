# Database Migrations

This directory contains Alembic database migrations for Forecastly.

## Quick Start

```bash
# Apply all pending migrations
alembic upgrade head

# Rollback the last migration
alembic downgrade -1

# Check current migration version
alembic current

# View migration history
alembic history --verbose
```

## What are Migrations?

Database migrations are version-controlled changes to your database schema. They allow you to:
- Track changes to database structure over time
- Apply schema changes consistently across environments
- Rollback changes if something goes wrong
- Collaborate with team members on schema changes

## Migration Files

Migrations are stored in `alembic/versions/` directory. Each file represents a single migration:

- `001_add_performance_indexes.py` - Adds database indexes for query optimization

## Creating New Migrations

### Auto-generate from Model Changes

When you modify SQLAlchemy models in `src/db/models.py`:

```bash
# Generate migration automatically
alembic revision --autogenerate -m "Add user profile table"

# Review the generated migration file
# Edit if necessary (auto-generation isn't perfect)

# Apply the migration
alembic upgrade head
```

### Manual Migration

For data migrations or custom SQL:

```bash
# Create empty migration file
alembic revision -m "Migrate legacy data"

# Edit the file and add upgrade/downgrade logic
# Apply the migration
alembic upgrade head
```

## Migration Workflow

### Development

1. **Make Model Changes**
   ```python
   # src/db/models.py
   class NewTable(Base):
       __tablename__ = 'new_table'
       id = Column(Integer, primary_key=True)
       name = Column(String(100))
   ```

2. **Generate Migration**
   ```bash
   alembic revision --autogenerate -m "Add new table"
   ```

3. **Review Generated Code**
   ```python
   # alembic/versions/002_add_new_table.py
   def upgrade():
       op.create_table('new_table',
           sa.Column('id', sa.Integer(), nullable=False),
           sa.Column('name', sa.String(100), nullable=True),
           sa.PrimaryKeyConstraint('id')
       )

   def downgrade():
       op.drop_table('new_table')
   ```

4. **Test Migration**
   ```bash
   # Apply
   alembic upgrade head

   # Test rollback
   alembic downgrade -1

   # Re-apply
   alembic upgrade head
   ```

5. **Commit to Git**
   ```bash
   git add alembic/versions/002_add_new_table.py
   git add src/db/models.py
   git commit -m "Add new table for feature X"
   ```

### Production Deployment

1. **Backup Database**
   ```bash
   # PostgreSQL
   pg_dump -U postgres forecastly > backup_before_migration.sql
   ```

2. **Review Pending Migrations**
   ```bash
   alembic current
   alembic history
   ```

3. **Test on Staging**
   ```bash
   # On staging environment
   alembic upgrade head
   # Run tests
   pytest tests/
   ```

4. **Apply to Production**
   ```bash
   # On production
   alembic upgrade head
   ```

5. **Monitor**
   - Check application logs
   - Verify database performance
   - Monitor error rates

## Common Commands

### Upgrade/Downgrade

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade 001

# Upgrade one step forward
alembic upgrade +1

# Downgrade one step back
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade 001

# Downgrade to base (empty database)
alembic downgrade base
```

### Information

```bash
# Current version
alembic current

# Migration history
alembic history

# Verbose history with details
alembic history --verbose

# Show SQL without executing
alembic upgrade head --sql

# Show specific revision
alembic show 001
```

### Branches and Merging

```bash
# Create branch point
alembic revision -m "Branch point" --branch-label mybranch

# Merge branches
alembic merge -m "Merge branches" branch1 branch2
```

## Migration Best Practices

### 1. Always Review Auto-generated Migrations

Auto-generation is helpful but not perfect:

```python
# ❌ Auto-generated might miss this
# Review and add manually if needed
op.create_index('ix_users_email_active',
                'users',
                ['email', 'is_active'],
                unique=False)
```

### 2. Make Migrations Reversible

Always implement both `upgrade()` and `downgrade()`:

```python
def upgrade():
    op.add_column('users', sa.Column('phone', sa.String(20)))

def downgrade():
    op.drop_column('users', 'phone')
```

### 3. Use Batch Operations for Large Tables

For tables with millions of rows:

```python
def upgrade():
    # Use CONCURRENTLY for PostgreSQL (doesn't lock table)
    op.create_index('ix_predictions_date',
                    'predictions',
                    ['date'],
                    postgresql_concurrently=True)
```

### 4. Data Migrations

When migrating data, consider:

```python
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Add new column
    op.add_column('users', sa.Column('full_name', sa.String(200)))

    # Migrate data
    connection = op.get_bind()
    connection.execute(
        sa.text("""
            UPDATE users
            SET full_name = CONCAT(first_name, ' ', last_name)
            WHERE full_name IS NULL
        """)
    )

    # Make column non-nullable
    op.alter_column('users', 'full_name', nullable=False)
```

### 5. Test Rollbacks

Always test that downgrade works:

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

### 6. Use Transactions

Migrations run in transactions by default. If needed:

```python
# Disable transaction for specific operations
def upgrade():
    # PostgreSQL CONCURRENTLY requires no transaction
    connection = op.get_bind()
    connection.execute('COMMIT')
    connection.execute(
        'CREATE INDEX CONCURRENTLY ix_name ON table(column)'
    )
```

## Troubleshooting

### Migration Fails

```bash
# Check current state
alembic current

# Mark as applied without running (dangerous!)
alembic stamp head

# Show SQL that would be executed
alembic upgrade head --sql
```

### Database Out of Sync

```bash
# Generate migration from current database state
alembic revision --autogenerate -m "Sync database"

# Review and apply
alembic upgrade head
```

### Merge Conflicts

```bash
# If two developers created migrations simultaneously
alembic merge -m "Merge parallel migrations" head1 head2
```

### Reset Alembic

```bash
# Drop alembic_version table
psql -U postgres forecastly -c "DROP TABLE alembic_version;"

# Re-initialize (marks all migrations as applied)
alembic stamp head
```

## Configuration

### alembic.ini

Main configuration file. Key settings:

```ini
[alembic]
script_location = %(here)s/alembic
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(second).2d_%%(rev)s_%%(slug)s
# Database URL is set programmatically in env.py
```

### env.py

Python environment configuration:

```python
# Imports project models
from src.db.models import Base

# Sets database URL from config
database_url = PATHS.get('database', {}).get('url')
config.set_main_option('sqlalchemy.url', database_url)

# Sets metadata for auto-generation
target_metadata = Base.metadata
```

## Multi-Environment Setup

### Development

```bash
# Use local database
export DATABASE_URL="postgresql://localhost/forecastly_dev"
alembic upgrade head
```

### Staging

```bash
# Use staging database
export DATABASE_URL="postgresql://staging.example.com/forecastly"
alembic upgrade head
```

### Production

```bash
# Use production database (from .env)
alembic upgrade head
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Database Migrations
  env:
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
  run: |
    alembic upgrade head
```

### Pre-deployment Check

```bash
#!/bin/bash
# Check for pending migrations
CURRENT=$(alembic current | grep -o '[a-f0-9]\{12\}')
HEAD=$(alembic heads | grep -o '[a-f0-9]\{12\}')

if [ "$CURRENT" != "$HEAD" ]; then
    echo "⚠️  Pending migrations detected!"
    alembic history
    exit 1
fi
```

## Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Database Indexes Guide](../docs/database_indexes.md)
- [PostgreSQL Migration Best Practices](https://www.postgresql.org/docs/current/ddl-alter.html)
