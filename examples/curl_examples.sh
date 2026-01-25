#!/bin/bash
# ==============================================================================
# Forecastly API - cURL Examples
# ==============================================================================
# Collection of curl commands to interact with Forecastly API
# ==============================================================================

# Configuration
API_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Forecastly API - cURL Examples${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ==============================================================================
# 1. Health Check
# ==============================================================================

echo -e "${GREEN}1. Health Check${NC}"
echo -e "${YELLOW}GET /health${NC}"
echo ""

curl -s "$API_URL/health" | jq '.'

echo ""
echo ""

# ==============================================================================
# 2. Get API Information
# ==============================================================================

echo -e "${GREEN}2. API Information${NC}"
echo -e "${YELLOW}GET /${NC}"
echo ""

curl -s "$API_URL/" | jq '.'

echo ""
echo ""

# ==============================================================================
# 3. Get Available SKUs
# ==============================================================================

echo -e "${GREEN}3. Get Available SKUs${NC}"
echo -e "${YELLOW}GET /api/v1/skus${NC}"
echo ""

curl -s "$API_URL/api/v1/skus" | jq '.skus[0:5]'

echo ""
echo ""

# ==============================================================================
# 4. Get Forecast for Specific SKU
# ==============================================================================

echo -e "${GREEN}4. Get Forecast for SKU001 (14 days)${NC}"
echo -e "${YELLOW}GET /api/v1/predict?sku_id=SKU001&horizon=14${NC}"
echo ""

curl -s "$API_URL/api/v1/predict?sku_id=SKU001&horizon=14" | jq '.predictions[0:3]'

echo ""
echo ""

# ==============================================================================
# 5. Get Forecast with Different Horizon
# ==============================================================================

echo -e "${GREEN}5. Get Forecast for SKU002 (7 days)${NC}"
echo -e "${YELLOW}GET /api/v1/predict?sku_id=SKU002&horizon=7${NC}"
echo ""

curl -s "$API_URL/api/v1/predict?sku_id=SKU002&horizon=7" | jq '{sku_id, horizon, count, source}'

echo ""
echo ""

# ==============================================================================
# 6. Get Model Metrics
# ==============================================================================

echo -e "${GREEN}6. Get Model Performance Metrics${NC}"
echo -e "${YELLOW}GET /api/v1/metrics${NC}"
echo ""

curl -s "$API_URL/api/v1/metrics" | jq '.metrics[0:3]'

echo ""
echo ""

# ==============================================================================
# 7. Get System Status
# ==============================================================================

echo -e "${GREEN}7. Get System Status${NC}"
echo -e "${YELLOW}GET /api/v1/status${NC}"
echo ""

curl -s "$API_URL/api/v1/status" | jq '.'

echo ""
echo ""

# ==============================================================================
# 8. Rebuild Forecasts
# ==============================================================================

echo -e "${GREEN}8. Rebuild Forecasts (14 days)${NC}"
echo -e "${YELLOW}POST /api/v1/predict/rebuild?horizon=14${NC}"
echo ""
echo -e "${YELLOW}Note: This may take several minutes...${NC}"
echo ""

# Uncomment to actually rebuild
# curl -s -X POST "$API_URL/api/v1/predict/rebuild?horizon=14" | jq '.'

echo "Skipped (uncomment in script to run)"

echo ""
echo ""

# ==============================================================================
# 9. Download Forecast as CSV
# ==============================================================================

echo -e "${GREEN}9. Download Forecast as CSV${NC}"
echo -e "${YELLOW}GET /api/v1/predict?sku_id=SKU001&horizon=14${NC}"
echo ""

OUTPUT_FILE="forecast_SKU001.csv"

curl -s "$API_URL/api/v1/predict?sku_id=SKU001&horizon=14" | \
    jq -r '.predictions[] | [.date, .prophet, .xgb, .ensemble] | @csv' | \
    cat <(echo "date,prophet,xgb,ensemble") - > "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}✓ Forecast saved to: $OUTPUT_FILE${NC}"
    echo ""
    head -n 5 "$OUTPUT_FILE"
else
    echo -e "${YELLOW}⚠ Failed to save forecast${NC}"
fi

echo ""
echo ""

# ==============================================================================
# 10. Batch Processing - Get Multiple SKUs
# ==============================================================================

echo -e "${GREEN}10. Batch Processing - First 3 SKUs${NC}"
echo ""

# Get SKU list
SKUS=$(curl -s "$API_URL/api/v1/skus" | jq -r '.skus[0:3][]')

for sku in $SKUS; do
    echo -e "${BLUE}Processing $sku...${NC}"
    AVG_FORECAST=$(curl -s "$API_URL/api/v1/predict?sku_id=$sku&horizon=7" | \
        jq '.predictions | map(.ensemble) | add / length')

    echo -e "  Average forecast: $AVG_FORECAST units/day"
done

echo ""
echo ""

# ==============================================================================
# 11. Error Handling Example - Invalid SKU
# ==============================================================================

echo -e "${GREEN}11. Error Handling - Invalid SKU${NC}"
echo -e "${YELLOW}GET /api/v1/predict?sku_id=INVALID${NC}"
echo ""

curl -s "$API_URL/api/v1/predict?sku_id=INVALID" | jq '.'

echo ""
echo ""

# ==============================================================================
# 12. Error Handling Example - Invalid Horizon
# ==============================================================================

echo -e "${GREEN}12. Error Handling - Invalid Horizon${NC}"
echo -e "${YELLOW}GET /api/v1/predict?sku_id=SKU001&horizon=999${NC}"
echo ""

curl -s "$API_URL/api/v1/predict?sku_id=SKU001&horizon=999" | jq '.'

echo ""
echo ""

# ==============================================================================
# 13. Export All Metrics to CSV
# ==============================================================================

echo -e "${GREEN}13. Export All Metrics to CSV${NC}"
echo ""

METRICS_FILE="metrics_export.csv"

curl -s "$API_URL/api/v1/metrics" | \
    jq -r '.metrics[] | [.sku_id, .mape_prophet, .mape_xgboost, .mape_ens, .best_model] | @csv' | \
    cat <(echo "sku_id,mape_prophet,mape_xgboost,mape_ens,best_model") - > "$METRICS_FILE"

if [ -f "$METRICS_FILE" ]; then
    echo -e "${GREEN}✓ Metrics saved to: $METRICS_FILE${NC}"
    echo ""
    head -n 5 "$METRICS_FILE"
else
    echo -e "${YELLOW}⚠ Failed to save metrics${NC}"
fi

echo ""
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Examples Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_FILE"
echo "  - $METRICS_FILE"
echo ""
