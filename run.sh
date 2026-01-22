#!/bin/bash

# Скрипт для быстрого запуска Forecastly проекта
# Использование: bash run.sh [option]
# опции: dev, docker, api, dashboard, logs, stop

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function show_help() {
    cat << EOF
${BLUE}Forecastly - Скрипт для быстрого запуска${NC}

Использование:
  bash run.sh [option]

Опции:
  dev          - Запуск в режиме разработки (локально)
  docker       - Запуск всех сервисов через Docker Compose
  api          - Запуск только API сервера
  dashboard    - Запуск только Streamlit дашборда
  logs         - Просмотр логов Docker контейнеров
  stop         - Остановка всех Docker контейнеров
  clean        - Очистка Docker образов и контейнеров
  help         - Показать эту справку

Примеры:
  bash run.sh dev         # Локальный запуск
  bash run.sh docker      # Запуск в Docker (рекомендуется)
  bash run.sh api         # Запуск только API
  bash run.sh logs        # Просмотр логов
EOF
}

function setup_venv() {
    if [ ! -d "venv" ]; then
        print_header "Создание виртуального окружения"
        python -m venv venv
        print_success "Виртуальное окружение создано"
    fi
    
    print_header "Активация виртуального окружения"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    print_success "Виртуальное окружение активировано"
}

function install_deps() {
    print_header "Установка зависимостей"
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Зависимости установлены"
}

function run_dev() {
    print_header "Запуск в режиме разработки"
    setup_venv
    install_deps
    
    echo -e "${YELLOW}Запуск Streamlit дашборда в отдельной сессии...${NC}"
    echo -e "${YELLOW}Затем запустите в другом терминале:${NC}"
    echo -e "${BLUE}source venv/bin/activate${NC}"
    echo -e "${BLUE}uvicorn src.api.app:app --reload --port 8000${NC}"
    
    streamlit run src/ui/dashboard.py
}

function run_docker() {
    print_header "Запуск Docker Compose"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен!"
        echo "Установи Docker Desktop или Docker Compose:"
        echo "https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_warning "Запуск всех сервисов (API, Dashboard, PostgreSQL)..."
    docker-compose up -d
    
    sleep 5
    
    print_header "Статус сервисов"
    docker-compose ps
    
    echo ""
    print_success "Все сервисы запущены!"
    echo ""
    echo -e "${BLUE}Доступные сервисы:${NC}"
    echo -e "  API:        ${GREEN}http://localhost:8000${NC}"
    echo -e "  API Docs:   ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  Dashboard:  ${GREEN}http://localhost:8501${NC}"
    echo -e "  Database:   ${GREEN}localhost:5432${NC}"
    echo ""
    echo "Просмотр логов:"
    echo "  docker-compose logs -f api"
    echo "  docker-compose logs -f dashboard"
    echo ""
    echo "Остановка:"
    echo "  docker-compose down"
}

function run_api() {
    print_header "Запуск API сервера"
    setup_venv
    install_deps
    
    echo ""
    print_success "Запуск API на http://localhost:8000"
    print_success "Документация: http://localhost:8000/docs"
    echo ""
    
    uvicorn src.api.app:app --reload --port 8000 --host 0.0.0.0
}

function run_dashboard() {
    print_header "Запуск Streamlit Dashboard"
    setup_venv
    install_deps
    
    echo ""
    print_success "Запуск Dashboard на http://localhost:8501"
    echo ""
    
    streamlit run src/ui/dashboard.py
}

function show_logs() {
    print_header "Просмотр логов Docker контейнеров"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен!"
        exit 1
    fi
    
    echo "Доступные сервисы:"
    docker-compose ps --services
    
    echo ""
    echo "Выбери сервис для просмотра логов (или Ctrl+C для выхода):"
    docker-compose logs -f
}

function stop_docker() {
    print_header "Остановка Docker контейнеров"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен!"
        exit 1
    fi
    
    docker-compose down
    print_success "Все контейнеры остановлены"
}

function clean_docker() {
    print_header "Очистка Docker образов и контейнеров"
    print_warning "Это удалит все неиспользуемые образы и контейнеры!"
    
    read -p "Продолжить? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker system prune -a --volumes -f
        print_success "Docker очищен"
    else
        print_warning "Отменено"
    fi
}

# Основная логика
case "${1:-help}" in
    dev)
        run_dev
        ;;
    docker)
        run_docker
        ;;
    api)
        run_api
        ;;
    dashboard)
        run_dashboard
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_docker
        ;;
    clean)
        clean_docker
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Неизвестная опция: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
