#!/usr/bin/env bash
set -e  # Exit on error

echo "============================================"
echo "Forecastly Demo Mode"
echo "============================================"
echo ""
echo "Step 1/6: Creating synthetic data..."
python -m src.etl.create_synthetic

echo ""
echo "Step 2/6: Preparing dataset..."
python -m src.etl.prepare_dataset

echo ""
echo "Step 3/6: Training Prophet model..."
python -m src.models.train_prophet

echo ""
echo "Step 4/6: Training XGBoost model..."
python -m src.models.train_xgboost

echo ""
echo "Step 5/6: Training LightGBM model..."
python -m src.models.train_lightgbm

echo ""
echo "Step 6/6: Generating predictions..."
python -m src.models.predict --horizon 14

echo ""
echo "============================================"
echo "Demo setup complete!"
echo "Starting Streamlit dashboard..."
echo "============================================"
echo ""
streamlit run src/ui/dashboard.py
