# ==============================================================================
# Forecastly - Makefile
# ==============================================================================
# Упрощенные команды для разработки, тестирования и развертывания
# ==============================================================================

.PHONY: help install install-dev test lint format clean run-api run-dashboard docker-up docker-down

# Цвета для вывода
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python interpreter
PYTHON := python
PIP := pip

# ==============================================================================
# Help
# ==============================================================================

help: ## Показать это сообщение помощи
	@echo "$(BLUE)Forecastly - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "install" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "run|dev|serve" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "test|lint|format|check" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "docker" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Data & Models:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "data|model|etl" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "clean" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ==============================================================================
# Installation
# ==============================================================================

install: ## Установить основные зависимости
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Установить зависимости для разработки
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

install-test: ## Установить зависимости для тестирования
	@echo "$(BLUE)Installing test dependencies...$(NC)"
	$(PIP) install -e ".[test]"
	@echo "$(GREEN)✓ Test environment ready$(NC)"

# ==============================================================================
# Development - Run Services
# ==============================================================================

run-api: ## Запустить FastAPI сервер
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	@echo "$(YELLOW)API will be available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Docs: http://localhost:8000/docs$(NC)"
	uvicorn src.api.app:app --reload --port 8000

run-dashboard: ## Запустить Streamlit дашборд
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	@echo "$(YELLOW)Dashboard will be available at: http://localhost:8501$(NC)"
	streamlit run src/ui/dashboard.py

run-all: ## Запустить API и дашборд параллельно (требует tmux или screen)
	@echo "$(BLUE)Starting all services...$(NC)"
	@echo "$(YELLOW)Note: This requires tmux or run in separate terminals$(NC)"
	@echo "$(YELLOW)Run 'make run-api' in one terminal and 'make run-dashboard' in another$(NC)"

# ==============================================================================
# Testing
# ==============================================================================

test: ## Запустить все тесты
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-unit: ## Запустить только unit тесты
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -v -m unit

test-integration: ## Запустить только integration тесты
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m integration

test-api: ## Запустить только тесты API
	@echo "$(BLUE)Running API tests...$(NC)"
	pytest tests/test_api.py tests/test_auth.py -v

test-cov: ## Запустить тесты с coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

test-watch: ## Запустить тесты в режиме watch (при изменении файлов)
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/

# ==============================================================================
# Code Quality
# ==============================================================================

lint: ## Проверить код линтерами
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "$(YELLOW)Flake8...$(NC)"
	flake8 src/ tests/
	@echo "$(YELLOW)MyPy...$(NC)"
	mypy src/
	@echo "$(GREEN)✓ Linting passed$(NC)"

format: ## Отформатировать код
	@echo "$(BLUE)Formatting code...$(NC)"
	@echo "$(YELLOW)Black...$(NC)"
	black src/ tests/
	@echo "$(YELLOW)isort...$(NC)"
	isort src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Проверить форматирование без изменений
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/

check: format-check lint test ## Полная проверка кода (format + lint + test)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Собрать Docker образы
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Запустить все сервисы через Docker Compose
	@echo "$(BLUE)Starting services with Docker Compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Dashboard: http://localhost:8501$(NC)"

docker-down: ## Остановить Docker Compose сервисы
	@echo "$(BLUE)Stopping Docker Compose services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs: ## Показать логи Docker контейнеров
	docker-compose logs -f

docker-restart: docker-down docker-up ## Перезапустить Docker Compose сервисы

docker-clean: ## Удалить Docker образы и volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down -v --rmi all
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

# ==============================================================================
# Data & Models
# ==============================================================================

data-generate: ## Сгенерировать синтетические данные
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	$(PYTHON) -m src.etl.create_synthetic
	@echo "$(GREEN)✓ Synthetic data generated$(NC)"

data-etl: ## Запустить ETL процесс
	@echo "$(BLUE)Running ETL pipeline...$(NC)"
	$(PYTHON) -c "from src.etl.prepare_dataset import main; main('data/raw/sales_synth.csv')"
	@echo "$(GREEN)✓ ETL completed$(NC)"

model-train-prophet: ## Обучить Prophet модель
	@echo "$(BLUE)Training Prophet model...$(NC)"
	$(PYTHON) -m src.models.train_prophet
	@echo "$(GREEN)✓ Prophet model trained$(NC)"

model-train-xgboost: ## Обучить XGBoost модель
	@echo "$(BLUE)Training XGBoost model...$(NC)"
	$(PYTHON) -m src.models.train_xgboost
	@echo "$(GREEN)✓ XGBoost model trained$(NC)"

model-train-all: model-train-prophet model-train-xgboost ## Обучить все модели
	@echo "$(GREEN)✓ All models trained$(NC)"

model-predict: ## Сделать прогноз
	@echo "$(BLUE)Running prediction...$(NC)"
	$(PYTHON) -m src.models.predict --horizon 14
	@echo "$(GREEN)✓ Predictions generated$(NC)"

model-evaluate: ## Оценить качество моделей
	@echo "$(BLUE)Evaluating models...$(NC)"
	$(PYTHON) -m src.models.evaluate --horizon 14
	@echo "$(GREEN)✓ Evaluation completed$(NC)"

pipeline-full: data-generate data-etl model-train-all model-predict model-evaluate ## Полный пайплайн от данных до оценки
	@echo "$(GREEN)✓ Full pipeline completed!$(NC)"

# ==============================================================================
# Database
# ==============================================================================

db-init: ## Инициализировать базу данных
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -c "from src.db.init_db import init_database; init_database()"
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate: ## Создать миграцию Alembic
	@echo "$(BLUE)Creating migration...$(NC)"
	alembic revision --autogenerate -m "$(message)"

db-upgrade: ## Применить миграции
	@echo "$(BLUE)Applying migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations applied$(NC)"

db-downgrade: ## Откатить последнюю миграцию
	@echo "$(BLUE)Rolling back migration...$(NC)"
	alembic downgrade -1

# ==============================================================================
# Cleanup
# ==============================================================================

clean: ## Удалить временные файлы и кэши
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-data: ## Удалить сгенерированные данные и модели
	@echo "$(RED)Warning: This will delete all data and models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* data/models/* logs/*; \
		echo "$(GREEN)✓ Data cleaned$(NC)"; \
	fi

clean-all: clean clean-data docker-clean ## Полная очистка (файлы, данные, Docker)
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# ==============================================================================
# Documentation
# ==============================================================================

docs-serve: ## Запустить локальный сервер документации (если используется mkdocs)
	@echo "$(BLUE)Starting documentation server...$(NC)"
	mkdocs serve

docs-build: ## Собрать документацию
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

# ==============================================================================
# Security
# ==============================================================================

security-check: ## Проверить безопасность зависимостей
	@echo "$(BLUE)Checking security vulnerabilities...$(NC)"
	pip-audit
	bandit -r src/

# ==============================================================================
# Deployment
# ==============================================================================

deploy-prepare: check ## Подготовка к деплою (проверки)
	@echo "$(GREEN)✓ Ready for deployment$(NC)"

version: ## Показать версию проекта
	@echo "$(BLUE)Forecastly version:$(NC) $$(grep version pyproject.toml | head -1 | cut -d'"' -f2)"
