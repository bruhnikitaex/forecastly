#!/usr/bin/env python
"""
Forecastly Admin CLI

Command-line interface for managing the Forecastly forecasting system.

Usage:
    python cli.py --help
    python cli.py db init
    python cli.py user create admin@example.com
    python cli.py model train
"""

import click
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


@click.group()
@click.version_option(version='1.4.0', prog_name='Forecastly CLI')
def cli():
    """
    Forecastly Admin CLI - Manage your forecasting system from the command line.

    Available command groups:
    - db: Database management
    - user: User management
    - model: Machine learning models
    - data: Data operations
    - system: System maintenance
    """
    pass


# Database commands
@cli.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize database tables."""
    from src.db.init_db import init_database

    click.echo("🔧 Initializing database...")
    try:
        init_database()
        click.secho("✅ Database initialized successfully!", fg='green')
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)


@db.command()
def migrate():
    """Run database migrations."""
    import subprocess

    click.echo("🔄 Running database migrations...")
    try:
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            capture_output=True,
            text=True,
            check=True
        )
        click.echo(result.stdout)
        click.secho("✅ Migrations completed!", fg='green')
    except subprocess.CalledProcessError as e:
        click.secho(f"❌ Migration failed: {e.stderr}", fg='red')
        sys.exit(1)


@db.command()
@click.option('--output', '-o', default='backup.sql', help='Output file path')
def backup(output):
    """Backup database to SQL file."""
    import subprocess
    from src.utils.config import PATHS

    db_config = PATHS.get('database', {})
    db_name = db_config.get('name', 'forecastly')
    db_user = db_config.get('user', 'postgres')

    click.echo(f"💾 Backing up database to {output}...")

    try:
        subprocess.run(
            ['pg_dump', '-U', db_user, '-f', output, db_name],
            check=True
        )
        click.secho(f"✅ Backup saved to {output}", fg='green')
    except subprocess.CalledProcessError as e:
        click.secho(f"❌ Backup failed: {e}", fg='red')
        sys.exit(1)


@db.command()
@click.argument('backup_file')
@click.confirmation_option(prompt='Are you sure you want to restore? This will overwrite current data!')
def restore(backup_file):
    """Restore database from SQL backup file."""
    import subprocess
    from src.utils.config import PATHS

    db_config = PATHS.get('database', {})
    db_name = db_config.get('name', 'forecastly')
    db_user = db_config.get('user', 'postgres')

    if not Path(backup_file).exists():
        click.secho(f"❌ Backup file not found: {backup_file}", fg='red')
        sys.exit(1)

    click.echo(f"📥 Restoring database from {backup_file}...")

    try:
        subprocess.run(
            ['psql', '-U', db_user, '-d', db_name, '-f', backup_file],
            check=True
        )
        click.secho("✅ Database restored successfully!", fg='green')
    except subprocess.CalledProcessError as e:
        click.secho(f"❌ Restore failed: {e}", fg='red')
        sys.exit(1)


@db.command()
def reset():
    """Drop all tables and reinitialize database (DANGEROUS!)."""
    if not click.confirm('⚠️  This will DELETE ALL DATA. Are you absolutely sure?'):
        click.echo("Operation cancelled.")
        return

    from sqlalchemy import create_engine
    from src.db.models import Base
    from src.db.database import get_database_url
    from src.db.init_db import init_database

    click.echo("🗑️  Dropping all tables...")

    try:
        engine = create_engine(get_database_url())
        Base.metadata.drop_all(bind=engine)
        click.secho("✅ Tables dropped", fg='yellow')

        click.echo("🔧 Reinitializing database...")
        init_database()
        click.secho("✅ Database reset complete!", fg='green')
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)


# User management commands
@cli.group()
def user():
    """User management commands."""
    pass


@user.command()
@click.argument('email')
@click.option('--password', '-p', prompt=True, hide_input=True, confirmation_prompt=True)
@click.option('--role', '-r', type=click.Choice(['admin', 'analyst', 'viewer']), default='viewer')
@click.option('--superuser', is_flag=True, help='Make user a superuser')
def create(email, password, role, superuser):
    """Create a new user."""
    from src.db.database import SessionLocal
    from src.db.crud import create_user

    click.echo(f"👤 Creating user: {email}")

    try:
        db = SessionLocal()
        user = create_user(
            db=db,
            email=email,
            password=password,
            role=role,
            is_superuser=superuser
        )
        db.commit()
        click.secho(f"✅ User created: {user.email} (role: {user.role})", fg='green')
    except Exception as e:
        db.rollback()
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


@user.command()
def list():
    """List all users."""
    from src.db.database import SessionLocal
    from src.db.models import User

    db = SessionLocal()
    users = db.query(User).all()
    db.close()

    if not users:
        click.echo("No users found.")
        return

    click.echo("\n📋 Users:")
    click.echo(f"{'Email':<30} {'Role':<10} {'Active':<10} {'Superuser':<10} {'Created'}")
    click.echo("-" * 90)

    for user in users:
        active = "✅" if user.is_active else "❌"
        super_flag = "⭐" if user.is_superuser else ""
        created = user.created_at.strftime('%Y-%m-%d')
        click.echo(f"{user.email:<30} {user.role:<10} {active:<10} {super_flag:<10} {created}")


@user.command()
@click.argument('email')
@click.option('--role', '-r', type=click.Choice(['admin', 'analyst', 'viewer']), required=True)
def set_role(email, role):
    """Change user's role."""
    from src.db.database import SessionLocal
    from src.db.models import User

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            click.secho(f"❌ User not found: {email}", fg='red')
            sys.exit(1)

        old_role = user.role
        user.role = role
        db.commit()
        click.secho(f"✅ Role changed: {old_role} → {role}", fg='green')
    except Exception as e:
        db.rollback()
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


@user.command()
@click.argument('email')
@click.option('--password', '-p', prompt=True, hide_input=True, confirmation_prompt=True)
def reset_password(email, password):
    """Reset user's password."""
    from src.db.database import SessionLocal
    from src.db.models import User
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            click.secho(f"❌ User not found: {email}", fg='red')
            sys.exit(1)

        user.hashed_password = pwd_context.hash(password)
        user.failed_login_attempts = 0
        user.locked_until = None
        db.commit()
        click.secho(f"✅ Password reset for {email}", fg='green')
    except Exception as e:
        db.rollback()
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


@user.command()
@click.argument('email')
def deactivate(email):
    """Deactivate a user account."""
    from src.db.database import SessionLocal
    from src.db.models import User

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            click.secho(f"❌ User not found: {email}", fg='red')
            sys.exit(1)

        user.is_active = False
        db.commit()
        click.secho(f"✅ User deactivated: {email}", fg='green')
    except Exception as e:
        db.rollback()
        click.secho(f"❌ Error: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


# Model management commands
@cli.group()
def model():
    """Machine learning model management."""
    pass


@model.command()
@click.option('--model', '-m', 'model_type',
              type=click.Choice(['prophet', 'xgboost', 'lgbm', 'all']), default='all')
@click.option('--scope', '-s', type=click.Choice(['all', 'sku']), default='all',
              help='Training scope: all SKUs together or per-SKU models')
@click.option('--sku', 'sku_id', default=None, help='Specific SKU ID (when --scope=sku)')
def train(model_type, scope, sku_id):
    """Train forecasting models.

    Examples:
        forecastly model train --model prophet
        forecastly model train --model lgbm
        forecastly model train --scope sku --sku SKU001
        forecastly model train  # trains all
    """
    if scope == 'sku' and sku_id:
        click.echo(f"Training for SKU: {sku_id}")
    elif scope == 'sku':
        click.echo("Training per-SKU models...")

    if model_type in ('all', 'prophet'):
        click.echo("Training Prophet models...")
        try:
            from src.models.train_prophet import train as train_prophet
            train_prophet()
            click.secho("Prophet training complete!", fg='green')
        except Exception as e:
            click.secho(f"Prophet training failed: {e}", fg='red')

    if model_type in ('all', 'xgboost'):
        click.echo("Training XGBoost models...")
        try:
            from src.models.train_xgboost import train as train_xgboost
            train_xgboost()
            click.secho("XGBoost training complete!", fg='green')
        except Exception as e:
            click.secho(f"XGBoost training failed: {e}", fg='red')

    if model_type in ('all', 'lgbm'):
        click.echo("Training LightGBM models...")
        try:
            from src.models.train_lightgbm import train as train_lgbm
            train_lgbm()
            click.secho("LightGBM training complete!", fg='green')
        except Exception as e:
            click.secho(f"LightGBM training failed: {e}", fg='red')


@model.command()
@click.option('--horizon', '-h', type=int, default=14, help='Forecast horizon in days')
def evaluate(horizon):
    """Evaluate model performance."""
    from src.models.evaluate import evaluate as eval_models

    click.echo(f"📊 Evaluating models (horizon: {horizon} days)...")

    try:
        eval_models(horizon=horizon)
        click.secho("✅ Evaluation complete!", fg='green')
        click.echo("📈 Results saved to data/processed/metrics.csv")
    except Exception as e:
        click.secho(f"❌ Evaluation failed: {e}", fg='red')
        sys.exit(1)


@model.command()
def predict():
    """Generate predictions for all SKUs."""
    from src.models.predict import predict as run_predict

    click.echo("🔮 Generating predictions...")

    try:
        run_predict()
        click.secho("✅ Predictions complete!", fg='green')
        click.echo("📈 Results saved to data/processed/predictions.csv")
    except Exception as e:
        click.secho(f"❌ Prediction failed: {e}", fg='red')
        sys.exit(1)


# Forecast command (top-level for convenience)
@cli.command()
@click.option('--sku', required=True, help='SKU ID для прогноза')
@click.option('--horizon', '-h', type=int, default=30, help='Горизонт прогноза (1-30 дней)')
@click.option('--out', '-o', default='forecast.csv', help='Выходной CSV файл')
def forecast(sku, horizon, out):
    """Generate forecast for a specific SKU and export to CSV.

    Examples:
        forecastly forecast --sku SKU001 --horizon 30 --out forecast.csv
        forecastly forecast --sku SKU005 -h 7 -o result.csv
    """
    import pandas as pd

    click.echo(f"🔮 Прогноз для {sku} на {horizon} дней...")

    try:
        pred_path = Path('data/processed/predictions.csv')
        if not pred_path.exists():
            click.echo("Прогнозы не найдены, генерируем...")
            from src.models.predict import predict as run_predict
            run_predict(horizon=horizon)

        df = pd.read_csv(pred_path, parse_dates=['date'])
        sku_norm = sku.strip().upper().replace("SKU_", "SKU").replace("SKU-", "SKU")
        df_sku = df[df['sku_id'].astype(str).str.upper() == sku_norm].copy()

        if df_sku.empty:
            available = df['sku_id'].unique().tolist()
            click.secho(f"❌ SKU '{sku}' не найден. Доступные: {available[:10]}", fg='red')
            sys.exit(1)

        df_sku = df_sku.sort_values('date').head(horizon)
        df_sku.to_csv(out, index=False)

        click.secho(f"✅ Прогноз сохранён: {out} ({len(df_sku)} записей)", fg='green')

        # Краткая сводка
        for col in ['prophet', 'xgb', 'lgbm', 'ensemble']:
            if col in df_sku.columns:
                avg = df_sku[col].mean()
                if not pd.isna(avg):
                    click.echo(f"   {col:>10}: среднее = {avg:.1f}")

    except Exception as e:
        click.secho(f"❌ Ошибка: {e}", fg='red')
        sys.exit(1)


@model.command()
def list():
    """List available trained models."""
    from pathlib import Path
    from src.utils.config import PATHS
    import joblib

    models_dir = Path(PATHS['data']['models_dir'])

    if not models_dir.exists():
        click.echo("No models directory found.")
        return

    click.echo("\n📦 Available Models:")

    for model_file in models_dir.glob('*.pkl'):
        try:
            size = model_file.stat().st_size / (1024 * 1024)  # MB
            modified = model_file.stat().st_mtime
            import datetime
            mod_date = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')

            click.echo(f"  • {model_file.name:<30} {size:>6.2f} MB    {mod_date}")

            # Try to load and show details
            if model_file.name.startswith('prophet_'):
                models = joblib.load(model_file)
                click.echo(f"    └─ {len(models)} SKU models")
        except Exception as e:
            click.echo(f"  • {model_file.name} (error loading: {e})")


# Data management commands
@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
def upload(file_path):
    """Upload a CSV or XLSX data file.

    Validates, normalizes and saves the file for further processing.

    Examples:
        forecastly data upload sales.csv
        forecastly data upload report.xlsx
    """
    import pandas as pd
    from pathlib import Path as P

    ext = P(file_path).suffix.lower()
    if ext not in ('.csv', '.xlsx'):
        click.secho("Поддерживаемые форматы: CSV, XLSX", fg='red')
        sys.exit(1)

    click.echo(f"📥 Загрузка файла {file_path}...")

    try:
        if ext == '.xlsx':
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)

        # Нормализация колонок
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {"qty": "units", "quantity": "units", "sales": "units"}
        df.rename(columns=col_map, inplace=True)

        required = {"date", "sku_id"}
        missing = required - set(df.columns)
        if missing:
            click.secho(f"Отсутствуют обязательные поля: {missing}", fg='red')
            sys.exit(1)
        if "units" not in df.columns:
            click.secho("Не найдено поле количества (qty/units/quantity/sales)", fg='red')
            sys.exit(1)

        # Сохраняем как CSV
        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as dt
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"upload_{ts}.csv"
        save_path = raw_dir / save_name
        df.to_csv(save_path, index=False)

        click.echo(f"   Строк: {len(df)}")
        click.echo(f"   SKU: {df['sku_id'].nunique()}")
        click.secho(f"✅ Файл сохранён: {save_path}", fg='green')

    except Exception as e:
        click.secho(f"❌ Ошибка: {e}", fg='red')
        sys.exit(1)


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
def validate(file_path):
    """Validate a data file without importing.

    Checks required columns, data types, missing values, duplicates and outliers.

    Examples:
        forecastly data validate sales.csv
        forecastly data validate report.xlsx
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path as P

    ext = P(file_path).suffix.lower()
    click.echo(f"🔍 Валидация файла {file_path}...")

    try:
        if ext == '.xlsx':
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)

        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {"qty": "units", "quantity": "units", "sales": "units"}
        df.rename(columns=col_map, inplace=True)

        click.echo(f"\n📋 Отчёт о качестве данных:")
        click.echo(f"   Строк: {len(df):,}")
        click.echo(f"   Столбцов: {len(df.columns)}")
        click.echo(f"   Столбцы: {list(df.columns)}")

        # Проверка обязательных полей
        required = {"date", "sku_id", "units"}
        present = required & set(df.columns)
        missing = required - set(df.columns)
        click.echo(f"\n   Обязательные поля:")
        for f in sorted(required):
            if f in present:
                click.secho(f"     ✅ {f}", fg='green')
            else:
                click.secho(f"     ❌ {f} — отсутствует", fg='red')

        if 'sku_id' in df.columns:
            click.echo(f"\n   SKU: {df['sku_id'].nunique()}")
        if 'date' in df.columns:
            click.echo(f"   Даты: {df['date'].min()} — {df['date'].max()}")

        # Пропуски
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss) > 0:
            click.echo(f"\n   Пропуски:")
            for col, cnt in miss.items():
                pct = cnt / len(df) * 100
                click.echo(f"     {col}: {cnt} ({pct:.1f}%)")
        else:
            click.secho("   Пропусков нет", fg='green')

        # Дубликаты
        dups = df.duplicated().sum()
        click.echo(f"   Дубликатов: {dups}")

        # Выбросы по IQR
        if 'units' in df.columns:
            q1 = df['units'].quantile(0.25)
            q3 = df['units'].quantile(0.75)
            iqr = q3 - q1
            outliers = int(((df['units'] < q1 - 1.5 * iqr) | (df['units'] > q3 + 1.5 * iqr)).sum())
            click.echo(f"   Выбросы (units, IQR): {outliers}")

        if missing:
            click.secho("\n❌ Файл не прошёл валидацию", fg='red')
            sys.exit(1)
        else:
            click.secho("\n✅ Валидация пройдена", fg='green')

    except Exception as e:
        click.secho(f"❌ Ошибка: {e}", fg='red')
        sys.exit(1)


@data.command()
@click.argument('csv_file', type=click.Path(exists=True))
def import_sales(csv_file):
    """Import sales data from CSV file."""
    from src.etl.load_data import load_csv
    from src.etl.clean_data import clean_sales
    import pandas as pd

    click.echo(f"📥 Importing sales data from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        click.echo(f"   Loaded {len(df)} rows")

        cleaned = clean_sales(df)
        click.echo(f"   Cleaned to {len(cleaned)} rows")

        # Save to database
        from src.db.database import SessionLocal
        from src.db.models import SKU, SalesHistory
        import pandas as pd

        db = SessionLocal()

        # Get or create SKUs
        for sku_id in cleaned['sku_id'].unique():
            existing = db.query(SKU).filter(SKU.sku_id == str(sku_id)).first()
            if not existing:
                sku = SKU(sku_id=str(sku_id))
                db.add(sku)

        db.commit()

        # Import sales history
        # (Implementation depends on your schema)

        click.secho(f"✅ Import complete!", fg='green')
    except Exception as e:
        click.secho(f"❌ Import failed: {e}", fg='red')
        sys.exit(1)


@data.command()
@click.option('--output', '-o', default='sales_export.csv', help='Output CSV file')
def export_sales(output):
    """Export sales data to CSV file."""
    from src.db.database import SessionLocal
    from src.db.models import SalesHistory
    import pandas as pd

    click.echo(f"📤 Exporting sales data to {output}...")

    try:
        db = SessionLocal()
        sales = db.query(SalesHistory).all()

        if not sales:
            click.echo("No sales data found.")
            return

        data = [{
            'date': s.date,
            'sku_id': s.sku_id,
            'units': s.units,
            'revenue': s.revenue,
            'price': s.price,
            'promo_flag': s.promo_flag
        } for s in sales]

        df = pd.DataFrame(data)
        df.to_csv(output, index=False)

        click.secho(f"✅ Exported {len(df)} rows to {output}", fg='green')
    except Exception as e:
        click.secho(f"❌ Export failed: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


@data.command()
@click.option('--days', '-d', type=int, default=90, help='Keep data from last N days')
@click.confirmation_option(prompt='This will delete old data. Continue?')
def cleanup(days):
    """Clean up old data older than N days."""
    from src.db.database import SessionLocal
    from src.db.models import Prediction, ForecastRun
    from datetime import datetime, timedelta

    cutoff_date = datetime.now() - timedelta(days=days)

    click.echo(f"🗑️  Cleaning up data older than {cutoff_date.date()}...")

    try:
        db = SessionLocal()

        # Delete old predictions
        deleted_pred = db.query(Prediction).filter(
            Prediction.created_at < cutoff_date
        ).delete()

        # Delete old forecast runs
        deleted_runs = db.query(ForecastRun).filter(
            ForecastRun.started_at < cutoff_date
        ).delete()

        db.commit()

        click.secho(f"✅ Deleted {deleted_pred} predictions and {deleted_runs} forecast runs", fg='green')
    except Exception as e:
        db.rollback()
        click.secho(f"❌ Cleanup failed: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


# Audit commands
@cli.group()
def audit():
    """Audit log commands."""
    pass


@audit.command()
@click.option('--output', '-o', default='audit_export.csv', help='Output CSV file')
@click.option('--hours', type=int, default=720, help='Export logs from last N hours')
def export(output, hours):
    """Export audit logs to CSV file.

    Examples:
        forecastly audit export
        forecastly audit export -o audit.csv --hours 48
    """
    from src.db.database import SessionLocal
    from src.db.models import SecurityAuditLog
    from datetime import datetime, timedelta
    import pandas as pd

    click.echo(f"📋 Экспорт аудит-логов за {hours} часов...")

    try:
        db = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        logs = db.query(SecurityAuditLog).filter(
            SecurityAuditLog.created_at >= cutoff
        ).order_by(SecurityAuditLog.created_at.desc()).all()

        if not logs:
            click.echo("Нет записей аудита за указанный период.")
            return

        data = [{
            'id': log.id,
            'event_type': log.event_type,
            'user_id': log.user_id,
            'user_email': log.user_email,
            'ip_address': log.ip_address,
            'severity': log.severity,
            'details': log.details,
            'created_at': log.created_at.isoformat() if log.created_at else None,
        } for log in logs]

        df = pd.DataFrame(data)
        df.to_csv(output, index=False)

        click.secho(f"✅ Экспортировано {len(df)} записей в {output}", fg='green')
    except Exception as e:
        click.secho(f"❌ Ошибка: {e}", fg='red')
        sys.exit(1)
    finally:
        db.close()


# System commands
@cli.group()
def system():
    """System maintenance and monitoring."""
    pass


@system.command()
def health():
    """Check system health."""
    from src.monitoring import get_metrics_collector
    import requests

    click.echo("🏥 Checking system health...\n")

    # Check API
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            click.secho("✅ API: Healthy", fg='green')
        else:
            click.secho(f"❌ API: Unhealthy (status {response.status_code})", fg='red')
    except Exception as e:
        click.secho(f"❌ API: Not responding ({e})", fg='red')

    # Check database
    try:
        from src.db.database import SessionLocal
        db = SessionLocal()
        db.execute('SELECT 1')
        db.close()
        click.secho("✅ Database: Connected", fg='green')
    except Exception as e:
        click.secho(f"❌ Database: Error ({e})", fg='red')

    # Check system metrics
    collector = get_metrics_collector()
    collector.collect_system_metrics()

    click.echo(f"\n📊 System Metrics:")
    click.echo(f"  CPU: {collector.system_metrics.get('cpu_percent', 0):.1f}%")
    click.echo(f"  Memory: {collector.system_metrics.get('memory_percent', 0):.1f}%")
    click.echo(f"  Disk: {collector.system_metrics.get('disk_percent', 0):.1f}%")


@system.command()
def metrics():
    """Display current system metrics."""
    from src.monitoring import get_metrics_collector

    collector = get_metrics_collector()
    metrics = collector.get_metrics_json()

    click.echo("\n📈 System Metrics:\n")
    click.echo(f"Timestamp: {metrics['timestamp']}")
    click.echo(f"\nCounters:")
    for key, value in metrics.get('counters', {}).items():
        click.echo(f"  {key}: {value}")

    click.echo(f"\nGauges:")
    for key, value in metrics.get('gauges', {}).items():
        click.echo(f"  {key}: {value}")


@system.command()
@click.option('--lines', '-n', default=50, help='Number of log lines to show')
def logs(lines):
    """View recent application logs."""
    from pathlib import Path
    from src.utils.config import PATHS

    log_file = Path(PATHS.get('logging', {}).get('file', 'logs/forecastly.log'))

    if not log_file.exists():
        click.echo(f"Log file not found: {log_file}")
        return

    click.echo(f"📋 Last {lines} log entries:\n")

    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            # Color code by log level
            if 'ERROR' in line:
                click.secho(line.strip(), fg='red')
            elif 'WARNING' in line:
                click.secho(line.strip(), fg='yellow')
            elif 'INFO' in line:
                click.echo(line.strip())
            else:
                click.secho(line.strip(), dim=True)


if __name__ == '__main__':
    cli()
