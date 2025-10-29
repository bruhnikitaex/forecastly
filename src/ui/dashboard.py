import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title='Forecastly', layout='wide')

# === Пути к данным ===
data_raw = Path('data/raw')
data_proc = Path('data/processed')
data_models = Path('data/models')
for p in [data_raw, data_proc, data_models]:
    p.mkdir(parents=True, exist_ok=True)

st.sidebar.title("Forecastly")
st.sidebar.caption("Система анализа и прогнозирования продаж")
st.sidebar.markdown("**Автор:** Вульферт Никита Евгеньевич  \n**Группа:** 122 ИСП")
st.sidebar.divider()

st.title('Система анализа и прогнозирования продаж')
st.caption('ETL → Аналитика → Прогнозирование')

tabs = st.tabs(["📊 Данные", "📈 Прогноз", "📐 Аналитика", "⚙️ Модели", "🧮 Метрики"])

# =====================================================================
# 📊 ДАННЫЕ
# =====================================================================
with tabs[0]:
    st.subheader("Работа с данными")
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Сгенерировать синтетические данные'):
            os.system('python -m src.etl.create_synthetic')
            st.success('✅ Синтетические данные созданы: data/raw/sales_synth.csv')
    with c2:
        if st.button('Запустить ETL (очистка + фичи)'):
            os.system('python -c "from src.etl.prepare_dataset import main; main(\'data/raw/sales_synth.csv\')"')
            st.success('✅ ETL завершён: data/processed/*')

    raw_path = data_raw / 'sales_synth.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=['date'])
        st.write(f'Строк: {len(df):,} | SKU: {df.sku_id.nunique()} | Магазинов: {df.store_id.nunique()}')
        st.dataframe(df.head(50), width='stretch')
    else:
        st.info('Нажми «Сгенерировать синтетические данные», чтобы получить датасет.')

# =====================================================================
# 📈 ПРОГНОЗ
# =====================================================================
with tabs[1]:
    st.subheader("Прогнозирование продаж")

    pred_path = data_proc / 'processed.parquet' / 'predictions.csv'
    df_raw = pd.read_csv(data_raw / 'sales_synth.csv', parse_dates=['date']) if (data_raw / 'sales_synth.csv').exists() else None

    if df_raw is None:
        st.info("Сначала сгенерируй данные во вкладке «Данные».")
    else:
        sku_list = df_raw['sku_id'].unique().tolist()
        c0, c1, c2 = st.columns([2,1,1])
        with c0:
            selected_sku = st.selectbox("Выберите товар (SKU)", sku_list)
        with c1:
            horizon = st.slider("Горизонт (дней)", 7, 60, 14)
        with c2:
            models_selected = st.multiselect("Модели", ["Ensemble", "Prophet", "LightGBM"],
                                             default=["Ensemble", "Prophet", "LightGBM"])

        if st.button("Сделать прогноз"):
            os.system(f"python -m src.models.predict --horizon {horizon}")
            st.success(f'✅ Прогноз на {horizon} дней выполнен!')

        if pred_path.exists():
            df_pred = pd.read_csv(pred_path, parse_dates=['date'])

            df_true = df_raw[df_raw['sku_id'] == selected_sku].copy()
            df_true_tail = df_true.sort_values('date').tail(120)

            if 'sku_id' in df_pred.columns:
                df_p = df_pred[df_pred['sku_id'] == selected_sku].copy()
            else:
                df_p = df_pred.copy()
                df_p['sku_id'] = selected_sku

            MODEL_COL = {"Prophet": "prophet", "LightGBM": "lgbm", "Ensemble": "ensemble"}
            color_map = {"prophet": "#00AEEF", "lgbm": "#F45B69", "ensemble": "#7AC74F"}

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.grid(True, alpha=0.25)
            ax.plot(df_true_tail['date'], df_true_tail['units'], label='Факт', color='black', linewidth=1.6)

            for name in models_selected:
                col = MODEL_COL.get(name)
                if col and col in df_p.columns:
                    ax.plot(df_p['date'], df_p[col], label=name, linewidth=2, color=color_map.get(col, None))

            ax.legend()
            ax.set_title(f"Прогноз продаж ({selected_sku})")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Продажи, шт.")
            st.pyplot(fig)

            st.dataframe(df_p.tail(20), width='stretch')
        else:
            st.info("Сначала сделай прогноз (кнопкой выше).")

# =====================================================================
# 📐 АНАЛИТИКА
# =====================================================================
with tabs[2]:
    st.subheader("Быстрая аналитика по продажам")
    raw_path = data_raw / 'sales_synth.csv'
    if not raw_path.exists():
        st.info("Сначала сгенерируй данные во вкладке «Данные».")
    else:
        df = pd.read_csv(raw_path, parse_dates=['date'])
        skus = df['sku_id'].unique().tolist()
        sku_a = st.selectbox("SKU для анализа", skus)
        tail = df[df['sku_id'] == sku_a].sort_values('date').tail(180)
        tail['rolling'] = tail['units'].rolling(14, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10,3.5))
        ax.plot(tail['date'], tail['units'], label='Факт', alpha=0.5)
        ax.plot(tail['date'], tail['rolling'], label='Тренд (14д)', linewidth=2)
        ax.legend()
        ax.set_title(f"Динамика продаж ({sku_a})")
        st.pyplot(fig)

# =====================================================================
# ⚙️ МОДЕЛИ
# =====================================================================
with tabs[3]:
    st.subheader("Обучение моделей")
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Обучить Prophet'):
            os.system('python -m src.models.train_prophet')
            st.success('✅ Prophet обучен!')
    with c2:
        if st.button('Обучить LightGBM'):
            os.system('python -m src.models.train_lgbm')
            st.success('✅ LightGBM обучен!')

# =====================================================================
# 🧮 МЕТРИКИ
# =====================================================================
with tabs[4]:
    st.subheader("Метрики и сравнение моделей")

    metrics_path = data_proc / 'metrics.csv'
    if st.button("Пересчитать метрики (14 дней)"):
        os.system("python -m src.models.evaluate --horizon 14")
        st.success('✅ Метрики пересчитаны!')

    if metrics_path.exists():
        met = pd.read_csv(metrics_path)
        st.dataframe(met, width='stretch')
        st.markdown("### Доля побед по MAPE")
        leaderboard = (
            met['best_model'].value_counts(normalize=True)
            .mul(100).round(1)
            .rename_axis('model').reset_index(name='%')
        )
        st.dataframe(leaderboard, width='stretch')
    else:
        st.info("Метрики пока не рассчитаны. Нажми кнопку выше.")
