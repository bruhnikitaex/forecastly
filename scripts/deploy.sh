#!/bin/bash
# ==============================================================================
# Forecastly - Deployment Script
# ==============================================================================
# Автоматическое развертывание в production
# ==============================================================================

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Forecastly - Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ==============================================================================
# Configuration
# ==============================================================================

ENVIRONMENT=${1:-production}  # production or staging
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}Deploying to: $ENVIRONMENT${NC}"
echo ""

# ==============================================================================
# Pre-deployment Checks
# ==============================================================================

echo -e "${YELLOW}Running pre-deployment checks...${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env file not found${NC}"
    echo "Please create .env from .env.example and configure it"
    exit 1
fi

# Check required environment variables
source .env

if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-change-me-in-production" ]; then
    echo -e "${RED}✗ SECRET_KEY not configured in .env${NC}"
    exit 1
fi

if [ "$USE_DATABASE" = "true" ] && [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}✗ POSTGRES_PASSWORD not configured in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment configuration OK${NC}"
echo ""

# ==============================================================================
# Code Quality Checks
# ==============================================================================

echo -e "${YELLOW}Running code quality checks...${NC}"

# Run tests
make test || {
    echo -e "${RED}✗ Tests failed${NC}"
    read -p "Continue deployment anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Run linters
make lint || {
    echo -e "${YELLOW}⚠ Linting warnings${NC}"
}

echo -e "${GREEN}✓ Code quality checks complete${NC}"
echo ""

# ==============================================================================
# Backup Current Deployment
# ==============================================================================

if [ -d "data" ] || [ -d "logs" ]; then
    echo -e "${YELLOW}Creating backup...${NC}"

    mkdir -p "$BACKUP_DIR"

    # Backup data and logs
    if [ -d "data" ]; then
        cp -r data "$BACKUP_DIR/"
    fi
    if [ -d "logs" ]; then
        cp -r logs "$BACKUP_DIR/"
    fi

    echo -e "${GREEN}✓ Backup created: $BACKUP_DIR${NC}"
    echo ""
fi

# ==============================================================================
# Docker Deployment
# ==============================================================================

echo -e "${YELLOW}Deploying with Docker Compose...${NC}"

# Stop running containers
docker-compose down

# Pull latest images (if using registry)
# docker-compose pull

# Build images
docker-compose build --no-cache

# Start services
docker-compose up -d

echo -e "${GREEN}✓ Services started${NC}"
echo ""

# ==============================================================================
# Health Checks
# ==============================================================================

echo -e "${YELLOW}Running health checks...${NC}"

# Wait for services to start
sleep 10

# Check API health
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is healthy${NC}"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -e "${YELLOW}Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}✗ API health check failed${NC}"
    echo "Check logs: docker-compose logs api"
    exit 1
fi

# Check Dashboard
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dashboard is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Dashboard health check failed${NC}"
fi

echo ""

# ==============================================================================
# Post-deployment Tasks
# ==============================================================================

echo -e "${YELLOW}Running post-deployment tasks...${NC}"

# Database migrations (if enabled)
if [ "$USE_DATABASE" = "true" ]; then
    docker-compose exec api alembic upgrade head || {
        echo -e "${YELLOW}⚠ Database migrations failed${NC}"
    }
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Services:${NC}"
echo "- API: http://localhost:8000"
echo "- Dashboard: http://localhost:8501"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "- docker-compose logs -f api  # View API logs"
echo "- docker-compose logs -f dashboard  # View dashboard logs"
echo "- docker-compose ps  # Check service status"
echo "- docker-compose down  # Stop services"
echo ""
echo -e "${YELLOW}Backup location:${NC}"
echo "$BACKUP_DIR"
echo ""
