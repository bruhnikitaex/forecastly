# src/ui/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# базовая настройка
# ---------------------------------------------------------

st.set_page_config(page_title='Forecastly', layout='wide')

# каталоги
data_raw = Path('data/raw')
data_proc = Path('data/processed')
data_models = Path('data/models')
logs_dir = Path('logs')

for p in [data_raw, data_proc, data_models, logs_dir]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# ФУНКЦИИ-ПРОВЕРКИ ДЛЯ СТАТУСОВ
# ---------------------------------------------------------
def has_raw() -> bool:
    return (data_raw / 'sales_synth.csv').exists() or any(data_raw.glob("*.csv")) or any(data_raw.glob("*.xlsx"))

def has_predictions() -> bool:
    return (data_proc / 'predictions.csv').exists()

def has_metrics() -> bool:
    return (data_proc / 'metrics.csv').exists()

def has_models() -> dict:
    return {
        "prophet": (data_models / 'prophet_model.pkl').exists(),
        "lgbm": (data_models / 'lgbm_model.pkl').exists()
    }

# ---------------------------------------------------------
# САЙДБАР
# ---------------------------------------------------------
st.sidebar.title("Forecastly")
st.sidebar.caption("Система анализа и прогнозирования продаж")
st.sidebar.markdown("**Автор:** Вульферт Никита Евгеньевич  \n**Группа:** 122 ИСП")
st.sidebar.divider()
st.sidebar.markdown("📁 Проект: `forecastly`")
st.sidebar.markdown("👨‍💻 Python + Streamlit + FastAPI")

# ---------------------------------------------------------
# ТИТУЛ
# ---------------------------------------------------------
st.title('Система анализа и прогнозирования продаж')
st.caption('ETL → Аналитика → Прогнозирование → Метрики')

# ---------------------------------------------------------
# СТАТУС-ПАНЕЛЬ
# ---------------------------------------------------------
st.markdown("### Статус системы")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**Данные**")
    if has_raw():
        st.success("есть")
    else:
        st.error("нет")

with c2:
    st.markdown("**Прогноз**")
    if has_predictions():
        st.success("есть")
    else:
        st.warning("нет")

with c3:
    st.markdown("**Метрики**")
    if has_metrics():
        st.success("есть")
    else:
        st.warning("нет")

with c4:
    st.markdown("**Модели**")
    models_state = has_models()
    txt = []
    if models_state["prophet"]:
        txt.append("Prophet ✅")
    else:
        txt.append("Prophet ❌")
    if models_state["lgbm"]:
        txt.append("LGBM ✅")
    else:
        txt.append("LGBM ❌")
    st.markdown("<br>".join(txt), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# ТАБЫ
# ---------------------------------------------------------
tabs = st.tabs(["📊 Данные", "📈 Прогноз", "📐 Аналитика", "⚙️ Модели", "🧮 Метрики"])

# =====================================================================
# 📊 1. ДАННЫЕ
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
            # по умолчанию берём синтетику
            os.system('python -c "from src.etl.prepare_dataset import main; main(\'data/raw/sales_synth.csv\')"')
            st.success('✅ ETL завершён: data/processed/*')

    st.markdown("#### Загрузить свой CSV с продажами")
    uploaded = st.file_uploader("Выбери CSV-файл с колонками: date, sku_id, store_id, units, price (price можно не указывать)", type=["csv"])
    if uploaded is not None:
        user_path = data_raw / "sales_user.csv"
        df_u = pd.read_csv(uploaded)
        df_u.to_csv(user_path, index=False)
        st.success(f"✅ Файл сохранён в {user_path}. Теперь можно запустить ETL именно на нём.")
        st.code(f"python -c \"from src.etl.prepare_dataset import main; main('data/raw/sales_user.csv')\"")

    raw_path = data_raw / 'sales_synth.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=['date'])
        st.write(f'Строк: {len(df):,} | SKU: {df.sku_id.nunique()} | Магазинов: {df.store_id.nunique()}')
        st.dataframe(df.head(50), width='stretch')
    else:
        st.info('Нажми «Сгенерировать синтетические данные», чтобы получить датасет или загрузи свой CSV.')

    st.divider()
    st.markdown("#### Логи последних запусков")
    log_path = logs_dir / "app.log"
    if log_path.exists():
        text = log_path.read_text(encoding='utf-8', errors='ignore')
        st.code(text[-2000:])
    else:
        st.caption("Логов пока нет. Они появятся после запуска ETL / обучения / предсказания.")

# =====================================================================
# 📈 2. ПРОГНОЗ
# =====================================================================
with tabs[1]:
    st.subheader("Прогнозирование продаж")

    pred_path = data_proc / 'predictions.csv'
    raw_path = data_raw / 'sales_synth.csv'
    df_raw = pd.read_csv(raw_path, parse_dates=['date']) if raw_path.exists() else None

    with st.expander("Файлы в data/processed"):
        if data_proc.exists():
            files = [str(p.relative_to(".")) for p in data_proc.rglob("*")]
            st.write(files if files else "папка пустая")
        else:
            st.write("папка data/processed не существует")

    if df_raw is None:
        st.info("Сначала сгенерируй/загрузи данные во вкладке «Данные».")
    else:
        stores = sorted(df_raw['store_id'].astype(str).unique().tolist())
        c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])

        with c0:
            selected_store = st.selectbox("Магазин", ["Все"] + stores)
        with c1:
            sku_list = df_raw['sku_id'].unique().tolist()
            selected_sku = st.selectbox("SKU", sku_list)
        with c2:
            horizon = st.slider("Горизонт (дн.)", 7, 60, 14)
        with c3:
            models_selected = st.multiselect(
                "Модели",
                ["Ensemble", "Prophet", "LightGBM"],
                default=["Ensemble", "Prophet", "LightGBM"]
            )

        if st.button("Сделать прогноз"):
            os.system(f"python -m src.models.predict --horizon {horizon}")
            st.success(f'✅ Прогноз на {horizon} дней выполнен!')

        if not pred_path.exists():
            st.info("Файл с прогнозом не найден. Нажми «Сделать прогноз».")
        else:
            df_pred = pd.read_csv(pred_path, parse_dates=['date'])

            # фактические данные
            df_true = df_raw[df_raw['sku_id'] == selected_sku].copy()
            if selected_store != "Все":
                df_true = df_true[df_true['store_id'] == selected_store]
            df_true_tail = df_true.sort_values('date').tail(120)

            # прогноз по SKU
            if 'sku_id' in df_pred.columns:
                df_p = df_pred[df_pred['sku_id'] == selected_sku].copy()
            else:
                df_p = df_pred.copy()
                df_p['sku_id'] = selected_sku

            if df_p.empty:
                st.warning("Прогноз есть, но по выбранному SKU строк нет. Скорее всего: разные форматы SKU. "
                           "Перегенерируй прогноз с обновлённым predict.py.")
                st.write("SKU, которые есть в прогнозе:", df_pred['sku_id'].unique().tolist())
            else:
                MODEL_COL = {"Prophet": "prophet", "LightGBM": "lgbm", "Ensemble": "ensemble"}
                color_map = {"prophet": "#00AEEF", "lgbm": "#F45B69", "ensemble": "#7AC74F"}

                fig, ax = plt.subplots(figsize=(11, 4))
                ax.grid(True, alpha=0.3)
                if not df_true_tail.empty:
                    ax.plot(df_true_tail['date'], df_true_tail['units'], label='Факт', color='black', linewidth=1.6)

                for name in models_selected:
                    col = MODEL_COL.get(name)
                    if col and col in df_p.columns:
                        ax.plot(df_p['date'], df_p[col], label=name, linewidth=2, color=color_map.get(col, None))

                ax.legend()
                ax.set_title(f"Прогноз продаж: {selected_sku} ({selected_store})")
                ax.set_xlabel("Дата")
                ax.set_ylabel("Продажи, шт.")
                st.pyplot(fig)

                st.markdown("#### Таблица прогноза")
                st.dataframe(df_p.tail(50), width='stretch')

                st.download_button(
                    "⬇️ Скачать прогноз (CSV)",
                    data=df_p.to_csv(index=False).encode('utf-8'),
                    file_name=f"forecast_{selected_sku}.csv",
                    mime="text/csv"
                )

            # показываем историю прогнозов, если есть
            hist_dir = data_proc / "history"
            if hist_dir.exists():
                st.markdown("#### История прогнозов")
                hist_files = sorted(hist_dir.glob("*.csv"))
                if hist_files:
                    for f in hist_files:
                        st.write(f"- {f.name}")
                else:
                    st.caption("История пока пустая.")
            else:
                st.caption("Папка data/processed/history пока не создана. Её создаёт predict.py при сохранении версии.")

    if log_path.exists() and log_path.stat().st_size > 2_000_000:
        log_path.unlink()


# =====================================================================
# 📐 3. АНАЛИТИКА
# =====================================================================
with tabs[2]:
    st.subheader("Быстрая аналитика по продажам")
    raw_path = data_raw / 'sales_synth.csv'
    if not raw_path.exists():
        st.info("Сначала сгенерируй данные.")
    else:
        df = pd.read_csv(raw_path, parse_dates=['date'])
        skus = df['sku_id'].unique().tolist()
        sku_a = st.selectbox("SKU для анализа", skus, key="anal_sku")
        tail = df[df['sku_id'] == sku_a].sort_values('date').tail(180)
        tail['rolling'] = tail['units'].rolling(14, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10,3.5))
        ax.plot(tail['date'], tail['units'], label='Факт', alpha=0.5)
        ax.plot(tail['date'], tail['rolling'], label='Тренд (14 дн.)', linewidth=2)
        ax.legend()
        ax.set_title(f"Динамика продаж ({sku_a})")
        st.pyplot(fig)

# =====================================================================
# ⚙️ 4. МОДЕЛИ
# =====================================================================
with tabs[3]:
    st.subheader("Обучение и управление моделями")

    c1, c2 = st.columns(2)
    with c1:
        if st.button('Обучить Prophet'):
            os.system('python -m src.models.train_prophet')
            st.success('✅ Prophet обучен!')
    with c2:
        if st.button('Обучить LightGBM'):
            os.system('python -m src.models.train_lgbm')
            st.success('✅ LightGBM обучен!')

    st.caption("После обучения можно перейти на вкладку «Прогноз» и пересчитать предсказания.")

# =====================================================================
# 🧮 5. МЕТРИКИ
# =====================================================================
with tabs[4]:
    st.subheader("Метрики, сравнение и паспорт модели")

    metrics_path = data_proc / 'metrics.csv'
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Пересчитать метрики (14 дней)"):
            os.system("python -m src.models.evaluate --horizon 14")
            st.success('✅ Метрики пересчитаны!')
    with c2:
        st.caption("Метрики считаются на основе ретро-прогноза по каждому SKU.")

    if metrics_path.exists():
        met = pd.read_csv(metrics_path)
        st.markdown("### Полная таблица метрик")
        st.dataframe(met, width='stretch')

        st.markdown("### Доля побед по MAPE")
        leaderboard = (
            met['best_model'].value_counts(normalize=True)
            .mul(100).round(1)
            .rename_axis('model').reset_index(name='%')
        )
        st.dataframe(leaderboard, width='stretch')

        st.divider()
        st.markdown("### Паспорт модели по SKU")
        sku_opts = met['sku_id'].unique().tolist()
        sel_sku = st.selectbox("Выбери SKU", sku_opts, key="passport_sku")
        row = met[met['sku_id'] == sel_sku].head(1)

        if not row.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Лучший алгоритм:** `{row['best_model'].iloc[0]}`")
                st.write(f"MAPE Prophet: {row['mape_prophet'].iloc[0]:.1f}%")
                st.write(f"MAPE LGBM: {row['mape_lgbm'].iloc[0]:.1f}%")
                st.write(f"MAPE Naive: {row['mape_naive'].iloc[0]:.1f}%")
                st.write(f"MAPE Ensemble: {row['mape_ens'].iloc[0]:.1f}%")
            with c2:
                labels = ['Prophet','LGBM','Naive','Ensemble']
                vals = [
                    row['mape_prophet'].iloc[0],
                    row['mape_lgbm'].iloc[0],
                    row['mape_naive'].iloc[0],
                    row['mape_ens'].iloc[0],
                ]
                fig2, ax2 = plt.subplots(figsize=(5,3))
                ax2.bar(labels, vals)
                ax2.set_ylabel("MAPE, %")
                st.pyplot(fig2)

        st.divider()
        st.download_button(
            "⬇️ Скачать metrics.csv",
            data=metrics_path.read_bytes(),
            file_name="metrics.csv",
            mime="text/csv"
        )

        # важность признаков
        st.divider()
        st.markdown("### Важность признаков LGBM")
        import joblib
        model_path = data_models / 'lgbm_model.pkl'
        if model_path.exists():
            try:
                lgbm = joblib.load(model_path)
                importances = getattr(lgbm, 'feature_importances_', None)
                names = getattr(lgbm, 'feature_name_', None)
                if importances is not None and names is not None:
                    fi = pd.DataFrame({'feature': names, 'importance': importances})
                    fi = fi.sort_values('importance', ascending=False).head(20)
                    figfi, axfi = plt.subplots(figsize=(8,5))
                    axfi.barh(fi['feature'][::-1], fi['importance'][::-1])
                    axfi.set_title("Top-20 feature importance (LGBM)")
                    st.pyplot(figfi)
                else:
                    st.info("У модели LGBM нет информации о важности признаков.")
            except Exception as e:
                st.warning(f"Не удалось загрузить LGBM: {e}")
        else:
            st.info("Модель LGBM ещё не обучена. Обучи её на вкладке «Модели».")
    else:
        st.info("Метрики пока не рассчитаны. Нажми «Пересчитать метрики».")
