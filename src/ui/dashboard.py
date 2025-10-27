import streamlit as st
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title='Sales Analytics & Forecast', layout='wide')

st.title('Система анализа и прогнозирования продаж')
st.caption('Демо: синтетика → ETL → модели → прогноз')

data_raw = Path('data/raw')
data_raw.mkdir(parents=True, exist_ok=True)
raw_path = data_raw / 'sales_synth.csv'

st.subheader('Действия')
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button('Сгенерировать синтетические данные'):
        os.system('python src/etl/generate_synthetic.py')
        st.success(f'Сгенерировано: {raw_path}')
with c2:
    if st.button('Запустить ETL (очистка+фичи)'):
        os.system('python -c "from src.etl.prepare_dataset import main; main()"')
        st.success('ETL завершён: data/processed/*')
with c3:
    if st.button('Обучить Prophet'):
        os.system('python src/models/train_prophet.py')
        st.success('Prophet обучен')
with c4:
    if st.button('Обучить LightGBM'):
        os.system('python src/models/train_lgbm.py')
        st.success('LightGBM обучен')

st.subheader('Данные')
if raw_path.exists():
    df = pd.read_csv(raw_path, parse_dates=['date'])
    st.write(f'Строк: {len(df):,} | SKU: {df.sku_id.nunique()} | Магазинов: {df.store_id.nunique()}')
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info('Нажми "Сгенерировать синтетические данные", чтобы получить демо-датасет.')

abc_path = Path('data/processed/abcxyz.csv')
if abc_path.exists():
    st.subheader('ABC/XYZ анализ SKU')
    abc = pd.read_csv(abc_path)
    st.dataframe(abc.head(50), use_container_width=True)
else:
    st.caption('После ETL появится таблица ABC/XYZ.')

pred_path = Path('data/processed/predictions.csv')
if pred_path.exists():
    st.subheader('Результаты прогноза')
    pred = pd.read_csv(pred_path, parse_dates=['date'])
    st.dataframe(pred, use_container_width=True)
else:
    st.caption('Сделай прогноз из раздела "Прогноз" или кнопками выше.')
