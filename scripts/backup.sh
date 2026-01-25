#!/bin/bash
# ==============================================================================
# Forecastly - Backup Script
# ==============================================================================
# Создает резервную копию данных, моделей и базы данных
# ==============================================================================

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="backups"
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Forecastly - Backup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ==============================================================================
# Create Backup Directory
# ==============================================================================

echo -e "${YELLOW}Creating backup directory...${NC}"
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓ Backup directory: $BACKUP_DIR${NC}"
echo ""

# ==============================================================================
# Backup Data Files
# ==============================================================================

if [ -d "data" ]; then
    echo -e "${YELLOW}Backing up data files...${NC}"

    # Backup data directory
    tar -czf "$BACKUP_DIR/data.tar.gz" data/

    # Get size
    SIZE=$(du -h "$BACKUP_DIR/data.tar.gz" | cut -f1)
    echo -e "${GREEN}✓ Data backed up ($SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ No data directory found${NC}"
fi

echo ""

# ==============================================================================
# Backup Models
# ==============================================================================

if [ -d "data/models" ]; then
    echo -e "${YELLOW}Backing up trained models...${NC}"

    # Backup models separately
    tar -czf "$BACKUP_DIR/models.tar.gz" data/models/

    SIZE=$(du -h "$BACKUP_DIR/models.tar.gz" | cut -f1)
    echo -e "${GREEN}✓ Models backed up ($SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ No models directory found${NC}"
fi

echo ""

# ==============================================================================
# Backup Database
# ==============================================================================

source .env 2>/dev/null || true

if [ "$USE_DATABASE" = "true" ]; then
    echo -e "${YELLOW}Backing up database...${NC}"

    # Determine if running in Docker
    if docker-compose ps | grep -q "forecastly-db"; then
        # Docker database backup
        docker-compose exec -T db pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_DIR/database.sql.gz"
    elif command -v pg_dump &> /dev/null; then
        # Local database backup
        PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_DIR/database.sql.gz"
    else
        echo -e "${YELLOW}⚠ pg_dump not available, skipping database backup${NC}"
    fi

    if [ -f "$BACKUP_DIR/database.sql.gz" ]; then
        SIZE=$(du -h "$BACKUP_DIR/database.sql.gz" | cut -f1)
        echo -e "${GREEN}✓ Database backed up ($SIZE)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Database mode disabled${NC}"
fi

echo ""

# ==============================================================================
# Backup Configuration
# ==============================================================================

echo -e "${YELLOW}Backing up configuration...${NC}"

# Backup .env (without sensitive data)
if [ -f ".env" ]; then
    grep -v "PASSWORD\|SECRET_KEY" .env > "$BACKUP_DIR/.env.safe" || true
fi

# Backup config files
if [ -d "configs" ]; then
    cp -r configs "$BACKUP_DIR/"
fi

echo -e "${GREEN}✓ Configuration backed up${NC}"
echo ""

# ==============================================================================
# Backup Logs
# ==============================================================================

if [ -d "logs" ]; then
    echo -e "${YELLOW}Backing up logs...${NC}"

    tar -czf "$BACKUP_DIR/logs.tar.gz" logs/

    SIZE=$(du -h "$BACKUP_DIR/logs.tar.gz" | cut -f1)
    echo -e "${GREEN}✓ Logs backed up ($SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ No logs directory found${NC}"
fi

echo ""

# ==============================================================================
# Create Backup Manifest
# ==============================================================================

echo -e "${YELLOW}Creating backup manifest...${NC}"

cat > "$BACKUP_DIR/manifest.txt" << EOF
Forecastly Backup Manifest
==========================
Timestamp: $TIMESTAMP
Date: $(date)
Hostname: $(hostname)

Contents:
EOF

ls -lh "$BACKUP_DIR" >> "$BACKUP_DIR/manifest.txt"

echo -e "${GREEN}✓ Manifest created${NC}"
echo ""

# ==============================================================================
# Cleanup Old Backups
# ==============================================================================

# Keep only last 7 backups
KEEP_COUNT=7

if [ -d "$BACKUP_ROOT" ]; then
    BACKUP_COUNT=$(ls -1 "$BACKUP_ROOT" | wc -l)

    if [ "$BACKUP_COUNT" -gt "$KEEP_COUNT" ]; then
        echo -e "${YELLOW}Cleaning up old backups (keeping last $KEEP_COUNT)...${NC}"

        ls -1t "$BACKUP_ROOT" | tail -n +$((KEEP_COUNT + 1)) | while read -r old_backup; do
            rm -rf "$BACKUP_ROOT/$old_backup"
            echo -e "  ${YELLOW}Removed: $old_backup${NC}"
        done

        echo -e "${GREEN}✓ Cleanup complete${NC}"
        echo ""
    fi
fi

# ==============================================================================
# Summary
# ==============================================================================

BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Backup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Backup location:${NC} $BACKUP_DIR"
echo -e "${YELLOW}Total size:${NC} $BACKUP_SIZE"
echo ""
echo -e "${YELLOW}Restore commands:${NC}"
echo "- Data: tar -xzf $BACKUP_DIR/data.tar.gz"
echo "- Models: tar -xzf $BACKUP_DIR/models.tar.gz"
echo "- Database: gunzip < $BACKUP_DIR/database.sql.gz | psql -U \$POSTGRES_USER \$POSTGRES_DB"
echo ""
