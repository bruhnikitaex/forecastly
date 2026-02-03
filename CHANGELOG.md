# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (No unreleased changes)

## [1.5.2] - 2026-02-03

### Added
- count_jobs() method for accurate job pagination
- .gitignore exception for .env.example template

### Changed
- Updated pyproject.toml version to 1.5.1
- Improved database error handling with specific SQLAlchemyError exceptions
- Enhanced transaction error logging in database.py

### Fixed
- SQLAlchemy anti-patterns: replaced == True/False with .is_(True/False)
- TODO in jobs_router.py: now returns actual total job count

## [1.5.1] - 2026-02-03

### Added
- .env.example file with comprehensive configuration template
- PostgreSQL import endpoint with security improvements
- Centralized version management in src/__version__.py
- XGBoost model training to demo scripts
- Comprehensive error handling with specific exception types
- Validation for PostgreSQL table and database names

### Changed
- Moved database credentials from Query parameters to POST body for security
- All datetime.now() calls updated to use timezone.utc
- Improved demo scripts (run_demo.bat/sh) with progress indicators and error handling
- Increased forecast horizon limit from 30 to 120 days
- Enhanced error messages with better logging

### Fixed
- **CRITICAL:** SQL Injection vulnerability in PostgreSQL import endpoint
- **CRITICAL:** Credentials exposure in Query parameters
- Duplicate "Выйти" button in Streamlit dashboard
- Missing timeout in subprocess.run() calls (added 300s timeout)
- Bare exception handlers replaced with specific exception types
- Hardcoded version string (now centralized)

### Security
- Added regex validation for table and database names to prevent SQL injection
- Password masking in logs
- Improved input validation across API endpoints

## [1.3.0] - 2026-01-26

### Added
- Comprehensive project documentation improvements
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Makefile for development commands
- pyproject.toml for modern Python package management
- CONTRIBUTING.md with contribution guidelines
- LICENSE file (MIT)
- Improved README with badges and troubleshooting section
- Complete API documentation with examples

### Changed
- Enhanced code quality tooling setup
- Improved development workflow
- Updated all version references to 1.3.0
- Updated year to 2026

### Fixed
- Fixed incorrect return type annotation in `predict.py`
- Fixed SQLAlchemy anti-patterns (`== True/False` replaced with `.is_(True/False)`)
- Fixed CLI imports for model training functions
- Fixed missing `datetime` import in `exceptions.py`
- Fixed `get_db_url` function reference in CLI (renamed to `get_database_url`)
- Added missing docstrings in `feature_builder.py`
- Fixed timestamp generation in exception handlers

## [1.1.0] - 2025-01-25

### Added
- Database support (PostgreSQL)
  - SQLAlchemy ORM models
  - Alembic migrations
  - CRUD operations for forecasts and metrics
- Authentication & Authorization
  - JWT token-based authentication
  - User registration and login
  - Password hashing with bcrypt
  - Role-based access control (admin/user)
- Security enhancements
  - Rate limiting (SlowAPI)
  - Account lockout after failed login attempts
  - Password policy enforcement
  - Audit logging for sensitive operations
- Database endpoints
  - `/api/v1/db/stats` - database statistics
  - `/api/v1/forecast-runs` - forecast run history
  - `/api/v1/db/sync` - sync CSV to database
- Docker improvements
  - Nginx reverse proxy configuration
  - PostgreSQL service in docker-compose
  - Health checks for all services
  - Production-ready HTTPS support (commented)

### Changed
- API version updated to 1.1.0
- Improved CORS configuration with environment variables
- Enhanced error handling in API endpoints
- Fallback to CSV when database is disabled
- Better logging throughout the application

### Fixed
- JSON serialization issues with NaN and Infinity values
- Date parsing in predictions endpoint
- SKU normalization consistency

## [1.0.0] - 2025-01-20

### Added
- Core forecasting functionality
  - Prophet model implementation
  - XGBoost model implementation
  - Ensemble forecasting (weighted average)
- ETL Pipeline
  - Data loading from CSV
  - Data cleaning and validation
  - Feature engineering
  - Synthetic data generation
  - ABC-XYZ analysis
- REST API (FastAPI)
  - `/health` - health check
  - `/api/v1/skus` - get available SKUs
  - `/api/v1/predict` - get forecast for SKU
  - `/api/v1/predict/rebuild` - rebuild forecasts
  - `/api/v1/metrics` - get model performance metrics
  - `/api/v1/status` - system status
- Streamlit Dashboard
  - Interactive data exploration
  - Real-time forecasting
  - Model training interface
  - Metrics visualization
  - Modern UI with custom CSS
- Model evaluation
  - MAPE calculation
  - Model comparison
  - Best model selection
  - Naive baseline
- Configuration management
  - YAML-based configuration (paths.yaml, model.yaml)
  - Environment variables support (.env)
  - Structured logging (Loguru)
- Testing
  - Comprehensive test suite with pytest
  - API endpoint tests
  - ETL pipeline tests
  - Model tests
  - Test fixtures and utilities
- Docker support
  - Dockerfile for API
  - Dockerfile for Streamlit
  - docker-compose.yml for orchestration
- Documentation
  - README.md with installation and usage
  - API documentation (docs/api.md)
  - Deployment guide (DEPLOYMENT.md)

### Technical Details
- Python 3.11+
- FastAPI for REST API
- Streamlit for dashboard
- Prophet & XGBoost for forecasting
- Pandas & NumPy for data processing
- Plotly & Matplotlib for visualization
- SQLAlchemy for ORM (v1.1.0+)
- PostgreSQL for database (v1.1.0+)

## [0.6.1] - 2025-01-15 (v0.6.1.dev0)

### Changed
- Model serialization improvements
- Bug fixes in Prophet model handling

## [0.6.0] - 2025-01-10

### Added
- Initial Prophet model integration
- Basic forecasting pipeline
- Data preprocessing utilities

## [0.1.0] - 2025-01-01

### Added
- Project initialization
- Basic project structure
- Initial dependencies setup

---

## Version History Summary

- **v1.3.0** (Current) - Bug fixes, code quality improvements
- **v1.1.0** - Database, Auth, Security
- **v1.0.0** - Core forecasting, API, Dashboard
- **v0.6.1** - Model improvements
- **v0.6.0** - Prophet integration
- **v0.1.0** - Project start

---

## Migration Guide

### Migrating from 1.0.0 to 1.1.0

1. **Environment Variables**
   - Add database configuration to `.env`:
     ```env
     USE_DATABASE=true
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     POSTGRES_DB=forecastly
     POSTGRES_USER=forecastly
     POSTGRES_PASSWORD=your_password
     SECRET_KEY=your_secret_key
     ```

2. **Database Setup**
   ```bash
   # Initialize database
   docker-compose up -d db
   python -c "from src.db.init_db import init_database; init_database()"

   # Sync existing data
   curl -X POST http://localhost:8000/api/v1/db/sync
   ```

3. **API Changes**
   - All endpoints now support both database and CSV modes
   - New authentication endpoints available at `/api/v1/auth/*`
   - Response format updated to include `source` field (database/csv)

4. **Breaking Changes**
   - None - backward compatible with 1.0.0

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/bruhnikita/forecastly/issues
- Documentation: [README.md](README.md)

---

**Автор:** Вульферт Никита Евгеньевич
**Группа:** 122 ИСП
**Год:** 2026
