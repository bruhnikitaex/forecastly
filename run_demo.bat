@echo off
echo ============================================
echo Forecastly Demo Mode
echo ============================================
echo.
echo Step 1/6: Creating synthetic data...
python -m src.etl.create_synthetic
if %errorlevel% neq 0 (
    echo ERROR: Failed to create synthetic data
    pause
    exit /b 1
)
echo.
echo Step 2/6: Preparing dataset...
python -m src.etl.prepare_dataset
if %errorlevel% neq 0 (
    echo ERROR: Failed to prepare dataset
    pause
    exit /b 1
)
echo.
echo Step 3/6: Training Prophet model...
python -m src.models.train_prophet
if %errorlevel% neq 0 (
    echo ERROR: Failed to train Prophet
    pause
    exit /b 1
)
echo.
echo Step 4/6: Training XGBoost model...
python -m src.models.train_xgboost
if %errorlevel% neq 0 (
    echo ERROR: Failed to train XGBoost
    pause
    exit /b 1
)
echo.
echo Step 5/6: Training LightGBM model...
python -m src.models.train_lightgbm
if %errorlevel% neq 0 (
    echo ERROR: Failed to train LightGBM
    pause
    exit /b 1
)
echo.
echo Step 6/6: Generating predictions...
python -m src.models.predict --horizon 14
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate predictions
    pause
    exit /b 1
)
echo.
echo ============================================
echo Demo setup complete!
echo Starting Streamlit dashboard...
echo ============================================
echo.
python -m streamlit run src/ui/dashboard.py
