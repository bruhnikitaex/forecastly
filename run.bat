@echo off
REM Скрипт для быстрого запуска Forecastly проекта (Windows)
REM Использование: run.bat [option]
REM опции: dev, docker, api, dashboard, logs, stop

setlocal enabledelayedexpansion

REM Определение цветов (приблизительно для Windows 10+)
set "RESET=[0m"
set "BLUE=[34m"
set "GREEN=[32m"
set "RED=[31m"
set "YELLOW=[33m"

if "%1"=="" (
    call :show_help
    exit /b 0
)

REM Переход в директорию проекта
cd /d "%~dp0"

:parse_args
if /i "%1"=="dev" (
    call :run_dev
    exit /b 0
)
if /i "%1"=="docker" (
    call :run_docker
    exit /b 0
)
if /i "%1"=="api" (
    call :run_api
    exit /b 0
)
if /i "%1"=="dashboard" (
    call :run_dashboard
    exit /b 0
)
if /i "%1"=="logs" (
    call :show_logs
    exit /b 0
)
if /i "%1"=="stop" (
    call :stop_docker
    exit /b 0
)
if /i "%1"=="clean" (
    call :clean_docker
    exit /b 0
)
if /i "%1"=="help" (
    call :show_help
    exit /b 0
)

echo Unknown option: %1
call :show_help
exit /b 1

:show_help
echo.
echo Forecastly - Скрипт для быстрого запуска
echo.
echo Использование:
echo   run.bat [option]
echo.
echo Опции:
echo   dev          - Запуск в режиме разработки (локально)
echo   docker       - Запуск всех сервисов через Docker Compose
echo   api          - Запуск только API сервера
echo   dashboard    - Запуск только Streamlit дашборда
echo   logs         - Просмотр логов Docker контейнеров
echo   stop         - Остановка всех Docker контейнеров
echo   clean        - Очистка Docker образов и контейнеров
echo   help         - Показать эту справку
echo.
echo Примеры:
echo   run.bat dev         - Локальный запуск
echo   run.bat docker      - Запуск в Docker (рекомендуется)
echo   run.bat api         - Запуск только API
echo   run.bat logs        - Просмотр логов
echo.
exit /b 0

:setup_venv
if not exist "venv" (
    echo.
    echo === Создание виртуального окружения ===
    python -m venv venv
    echo [OK] Виртуальное окружение создано
)

echo === Активация виртуального окружения ===
call venv\Scripts\activate.bat
echo [OK] Виртуальное окружение активировано
exit /b 0

:install_deps
echo === Установка зависимостей ===
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [OK] Зависимости установлены
exit /b 0

:run_dev
echo.
echo === Запуск в режиме разработки ===
call :setup_venv
call :install_deps
echo.
echo [WARNING] Запуск Streamlit дашборда...
echo.
echo Затем запустите в другом терминале (PowerShell/CMD):
echo   venv\Scripts\activate.bat
echo   uvicorn src.api.app:app --reload --port 8000
echo.
streamlit run src/ui/dashboard.py
exit /b 0

:run_docker
echo.
echo === Запуск Docker Compose ===

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose не установлен!
    echo.
    echo Установи Docker Desktop:
    echo https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [INFO] Запуск всех сервисов (API, Dashboard, PostgreSQL)...
docker-compose up -d

timeout /t 5 /nobreak

echo.
echo === Статус сервисов ===
docker-compose ps

echo.
echo [OK] Все сервисы запущены!
echo.
echo Доступные сервисы:
echo   API:        http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo   Dashboard:  http://localhost:8501
echo   Database:   localhost:5432
echo.
echo Просмотр логов:
echo   docker-compose logs -f api
echo   docker-compose logs -f dashboard
echo.
echo Остановка:
echo   docker-compose down
echo.
exit /b 0

:run_api
echo.
echo === Запуск API сервера ===
call :setup_venv
call :install_deps
echo.
echo [OK] Запуск API на http://localhost:8000
echo [OK] Документация: http://localhost:8000/docs
echo.
uvicorn src.api.app:app --reload --port 8000 --host 0.0.0.0
exit /b 0

:run_dashboard
echo.
echo === Запуск Streamlit Dashboard ===
call :setup_venv
call :install_deps
echo.
echo [OK] Запуск Dashboard на http://localhost:8501
echo.
streamlit run src/ui/dashboard.py
exit /b 0

:show_logs
echo.
echo === Просмотр логов Docker контейнеров ===

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose не установлен!
    exit /b 1
)

echo Доступные сервисы:
docker-compose ps --services
echo.
echo Просмотр логов (Ctrl+C для выхода):
docker-compose logs -f
exit /b 0

:stop_docker
echo.
echo === Остановка Docker контейнеров ===

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose не установлен!
    exit /b 1
)

docker-compose down
echo [OK] Все контейнеры остановлены
exit /b 0

:clean_docker
echo.
echo === Очистка Docker образов и контейнеров ===
echo [WARNING] Это удалит все неиспользуемые образы и контейнеры!
echo.

set /p confirm="Продолжить? (y/n): "
if /i not "%confirm%"=="y" (
    echo [INFO] Отменено
    exit /b 0
)

docker system prune -a --volumes -f
echo [OK] Docker очищен
exit /b 0

endlocal
