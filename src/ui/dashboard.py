import streamlit as st
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title='Sales Analytics & Forecast', layout='wide')
st.title('Система анализа и прогнозирования продаж')
st.caption('ETL → Аналитика → Прогнозирование')

# --- Пути к данным ---
data_raw = Path('data/raw')
data_proc = Path('data/processed')
data_models = Path('data/models')
data_raw.mkdir(parents=True, exist_ok=True)
data_proc.mkdir(parents=True, exist_ok=True)
data_models.mkdir(parents=True, exist_ok=True)

# --- Навигация ---
tabs = st.tabs(["📊 Данные", "📈 Прогноз", "⚙️ Модели", "📐 Метрики"])

# =====================================================================
# 📊 Вкладка 1. ДАННЫЕ
# =====================================================================
with tabs[0]:
    st.subheader("Работа с данными")

    c1, c2 = st.columns(2)
    with c1:
        if st.button('Сгенерировать синтетические данные'):
            os.system('python -m src.etl.create_synthetic')
            st.success('Синтетические данные созданы: data/raw/sales_synth.csv')
    with c2:
        if st.button('Запустить ETL (очистка + фичи)'):
            os.system('python -c "from src.etl.prepare_dataset import main; main(\'data/raw/sales_synth.csv\')"')
            st.success('ETL завершён: data/processed/*')

    raw_path = data_raw / 'sales_synth.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=['date'])
        st.write(f'Строк: {len(df):,} | SKU: {df.sku_id.nunique()} | Магазинов: {df.store_id.nunique()}')
        st.dataframe(df.head(50), width='stretch')
    else:
        st.info('Нажми «Сгенерировать синтетические данные», чтобы получить датасет.')

# =====================================================================
# 📈 Вкладка 2. ПРОГНОЗ
# =====================================================================
with tabs[1]:
    st.subheader("Прогнозирование продаж")

    pred_path = data_proc / 'predictions.csv'

    # загрузка исходных данных
    df_raw = None
    if (data_raw / 'sales_synth.csv').exists():
        df_raw = pd.read_csv(data_raw / 'sales_synth.csv', parse_dates=['date'])

    if df_raw is None:
        st.info("Сначала сгенерируй данные во вкладке «Данные».")
    else:
        sku_list = df_raw['sku_id'].unique().tolist()
        selected_sku = st.selectbox("Выберите товар (SKU)", sku_list)
        horizon = st.slider("Горизонт прогноза (дней)", 7, 60, 14)
        model_choice = st.multiselect(
            "Выберите модель",
            ["Prophet", "LightGBM", "Ensemble"],
            default=["Ensemble", "Prophet", "LightGBM"]
        )

        if st.button("Сделать прогноз"):
            os.system(f"python -m src.models.predict --horizon {horizon}")
            st.success(f'Прогноз на {horizon} дней выполнен!')

        if pred_path.exists():
            df_pred = pd.read_csv(pred_path, parse_dates=['date'])

            # фактические значения (последние 120 дней)
            df_true = df_raw[df_raw['sku_id'] == selected_sku].copy()
            df_true_tail = df_true.sort_values('date').tail(120)

            # безопасная фильтрация прогнозов по SKU
            if 'sku_id' in df_pred.columns:
                df_p = df_pred[df_pred['sku_id'] == selected_sku].copy()
            else:
                df_p = df_pred.copy()
                df_p['sku_id'] = selected_sku

            # === График (с корректным отображением моделей) ===
            MODEL_COL = {"Prophet": "prophet", "LightGBM": "lgbm", "Ensemble": "ensemble"}

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_true_tail['date'], df_true_tail['units'], label='Факт', color='black')

            for model in model_choice:
                col = MODEL_COL.get(model)
                if col and col in df_p.columns:
                    ax.plot(df_p['date'], df_p[col], label=model)

            ax.legend()
            ax.set_title(f"Прогноз продаж ({selected_sku})")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Продажи, шт.")
            st.pyplot(fig)

            st.dataframe(df_p.tail(20), width='stretch')
        else:
            st.info("Сначала сделай прогноз (кнопкой выше).")


# =====================================================================
# ⚙️ Вкладка 3. МОДЕЛИ
# =====================================================================
with tabs[2]:
    st.subheader("Управление моделями")

    c1, c2 = st.columns(2)
    with c1:
        if st.button('Обучить Prophet (по каждому SKU)'):
            os.system('python -m src.models.train_prophet')
            st.success('Модели Prophet обучены!')
    with c2:
        if st.button('Обучить LightGBM'):
            os.system('python -m src.models.train_lgbm')
            st.success('Модель LightGBM обучена!')

    st.caption("После обучения можно перейти на вкладку «Прогноз» для визуализации или «Метрики» для оценки точности.")

# =====================================================================
# 📐 Вкладка 4. МЕТРИКИ
# =====================================================================
with tabs[3]:
    st.subheader("Оценка точности моделей")

    metrics_path = data_proc / 'metrics.csv'
    raw_path = data_raw / 'sales_synth.csv'

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        horizon_eval = st.slider("Горизонт (дней)", 7, 60, 14, key="eval_hor")
    with c2:
        if st.button("Пересчитать метрики"):
            os.system(f'python -m src.models.evaluate --horizon {horizon_eval}')
            st.success(f'Метрики обновлены для горизонта {horizon_eval} дней')

    if not metrics_path.exists():
        st.info("Метрик пока нет. Нажми «Пересчитать метрики».")
    else:
        met = pd.read_csv(metrics_path)
        st.write(f"Строк: {len(met):,}")
        st.dataframe(met, width='stretch')

        st.markdown("**Доля лучших по MAPE (Prophet / LGBM / Naive):**")
        summary = (
            met['best_model']
            .value_counts(normalize=True)
            .mul(100).round(1)
            .rename_axis('model').reset_index(name='%')
        )
        st.dataframe(summary, width='stretch')

        if raw_path.exists():
            df_raw_ = pd.read_csv(raw_path)
            sku_list = df_raw_['sku_id'].unique().tolist()
            sel_sku = st.selectbox("Сравнение по SKU", sku_list, key="metric_sku")
            row = met[met['sku_id'] == sel_sku].head(1)
            if not row.empty:
                m_prophet = float(row['mape_prophet'].iloc[0]) if 'mape_prophet' in row else float('nan')
                m_lgbm   = float(row['mape_lgbm'].iloc[0]) if 'mape_lgbm' in row else float('nan')
                m_naive  = float(row['mape_naive'].iloc[0]) if 'mape_naive' in row else float('nan')

                st.caption(f"MAPE для {sel_sku}")
                fig, ax = plt.subplots(figsize=(5, 3))
                labels = ['Prophet', 'LGBM', 'Naive']
                values = [m_prophet, m_lgbm, m_naive]
                ax.bar(labels, values)
                ax.set_ylabel('MAPE, %')
                ax.set_ylim(0, max([v for v in values if pd.notna(v)] + [1]) * 1.2)
                for i, v in enumerate(values):
                    if pd.notna(v):
                        ax.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
                st.pyplot(fig)

        st.divider()
        st.download_button(
            label="⬇️ Скачать metrics.csv",
            data=metrics_path.read_bytes(),
            file_name="metrics.csv",
            mime="text/csv"
        )
        best_filter = st.selectbox("Экспорт по лучшей модели (фильтр)", ['all', 'prophet', 'lgbm', 'naive'])
        if best_filter != 'all':
            met_filtered = met[met['best_model'].str.lower() == best_filter.lower()]
            st.download_button(
                label=f"⬇️ Скачать metrics_{best_filter}.csv",
                data=met_filtered.to_csv(index=False).encode('utf-8'),
                file_name=f"metrics_{best_filter}.csv",
                mime="text/csv"
            )
