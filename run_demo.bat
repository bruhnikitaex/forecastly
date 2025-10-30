@echo off
python -m src.etl.create_synthetic
python -m src.etl.prepare_dataset
python -m src.models.train_prophet
python -m src.models.train_lgbm
python -m src.models.predict --horizon 14
python -m streamlit run src/ui/dashboard.py
