#!/bin/bash
# ==============================================================================
# Forecastly - Setup Script
# ==============================================================================
# Автоматическая настройка окружения для разработки
# ==============================================================================

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Forecastly - Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ==============================================================================
# Check Prerequisites
# ==============================================================================

echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3.11+ is required${NC}"
    exit 1
fi

# Check Git
if command -v git &> /dev/null; then
    echo -e "${GREEN}✓ Git installed${NC}"
else
    echo -e "${RED}✗ Git is required${NC}"
    exit 1
fi

# Check Docker (optional)
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker installed${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ Docker not found (optional)${NC}"
    DOCKER_AVAILABLE=false
fi

echo ""

# ==============================================================================
# Create Virtual Environment
# ==============================================================================

echo -e "${YELLOW}Creating virtual environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

echo ""

# ==============================================================================
# Install Dependencies
# ==============================================================================

echo -e "${YELLOW}Installing dependencies...${NC}"

pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# ==============================================================================
# Install Development Tools
# ==============================================================================

echo -e "${YELLOW}Installing development tools...${NC}"

pip install -e ".[dev]"
pre-commit install

echo -e "${GREEN}✓ Development tools installed${NC}"
echo ""

# ==============================================================================
# Create .env File
# ==============================================================================

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env

    # Generate secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i.bak "s/your-secret-key-change-me-in-production/$SECRET_KEY/" .env
    rm .env.bak 2>/dev/null || true

    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env and configure database credentials${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

echo ""

# ==============================================================================
# Create Data Directories
# ==============================================================================

echo -e "${YELLOW}Creating data directories...${NC}"

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs

echo -e "${GREEN}✓ Data directories created${NC}"
echo ""

# ==============================================================================
# Generate Synthetic Data
# ==============================================================================

read -p "Generate synthetic data for testing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Generating synthetic data...${NC}"
    python -m src.etl.create_synthetic
    echo -e "${GREEN}✓ Synthetic data generated${NC}"
fi

echo ""

# ==============================================================================
# Docker Setup (Optional)
# ==============================================================================

if [ "$DOCKER_AVAILABLE" = true ]; then
    read -p "Build Docker images? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker-compose build
        echo -e "${GREEN}✓ Docker images built${NC}"
    fi
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review and update .env file"
echo "2. Start API: make run-api"
echo "3. Start Dashboard: make run-dashboard"
echo "4. Run tests: make test"
echo ""
echo -e "${YELLOW}Docker commands:${NC}"
echo "- docker-compose up -d  # Start all services"
echo "- docker-compose logs -f  # View logs"
echo ""
echo "For more commands, run: make help"
echo ""
