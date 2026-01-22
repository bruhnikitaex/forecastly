"""
Streamlit дашборд для системы прогнозирования продаж Forecastly.

Интерактивный интерфейс с современным дизайном для:
- Загрузки и просмотра данных
- Выполнения ETL процесса
- Обучения моделей
- Просмотра прогнозов
- Анализа метрик качества
"""

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Настройка страницы
# ---------------------------------------------------------
st.set_page_config(
    page_title='Forecastly',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ---------------------------------------------------------
# Custom CSS для современного вида
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Основные стили */
    .main {
        padding: 1rem 2rem;
    }

    /* Заголовок */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* KPI карточки */
    .kpi-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Статус индикаторы */
    .status-ok {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }

    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }

    .status-error {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }

    /* Секции */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, transparent 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }

    /* Таблицы */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Кнопки */
    .stButton > button {
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Info boxes */
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #1976d2;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Success box */
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Каталоги
# ---------------------------------------------------------
data_raw = Path('data/raw')
data_proc = Path('data/processed')
data_models = Path('data/models')
logs_dir = Path('logs')

for p in [data_raw, data_proc, data_models, logs_dir]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Цветовая палитра
# ---------------------------------------------------------
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#38ef7d',
    'warning': '#f5576c',
    'prophet': '#00d4ff',
    'xgboost': '#ff6b6b',
    'ensemble': '#4ecdc4',
    'fact': '#2d3436',
    'background': '#f8f9fa'
}

# ---------------------------------------------------------
# Кэшированные функции
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_system_status():
    """Получает статус всех компонентов системы."""
    return {
        "raw_data": (data_raw / 'sales_synth.csv').exists() or any(data_raw.glob("*.csv")),
        "predictions": (data_proc / 'predictions.csv').exists(),
        "metrics": (data_proc / 'metrics.csv').exists(),
        "prophet": (data_models / 'prophet_model.pkl').exists(),
        "xgboost": (data_models / 'xgboost_model.pkl').exists()
    }

@st.cache_data
def load_raw_data(file_path: str = 'data/raw/sales_synth.csv') -> pd.DataFrame:
    """Загружает сырые данные с кэшированием."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df

@st.cache_data
def load_predictions_data(file_path: str = 'data/processed/predictions.csv') -> pd.DataFrame:
    """Загружает прогнозы с кэшированием."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df

@st.cache_data
def load_metrics_data(file_path: str = 'data/processed/metrics.csv') -> pd.DataFrame:
    """Загружает метрики с кэшированием."""
    return pd.read_csv(file_path)

def run_command(cmd: list, description: str) -> tuple:
    """Выполняет команду и возвращает результат."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr

# ---------------------------------------------------------
# САЙДБАР
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="color: #667eea; margin: 0;">📊 Forecastly</h1>
        <p style="color: #888; font-size: 0.9rem;">Sales Forecasting System</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Статус системы в сайдбаре
    status = get_system_status()

    st.markdown("### 🔄 Статус системы")

    col1, col2 = st.columns(2)
    with col1:
        if status["raw_data"]:
            st.success("Данные ✓")
        else:
            st.error("Данные ✗")

        if status["predictions"]:
            st.success("Прогноз ✓")
        else:
            st.warning("Прогноз ✗")

    with col2:
        if status["prophet"]:
            st.success("Prophet ✓")
        else:
            st.warning("Prophet ✗")

        if status["xgboost"]:
            st.success("XGBoost ✓")
        else:
            st.warning("XGBoost ✗")

    st.divider()

    # Быстрые действия
    st.markdown("### ⚡ Быстрые действия")

    if st.button("🔄 Обновить статус", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Информация
    st.markdown("### 📋 О проекте")
    st.markdown("""
    **Автор:** Вульферт Н.Е.
    **Группа:** 122 ИСП
    **Год:** 2025
    """)

    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
        <p style="margin: 0; font-size: 0.8rem; color: #667eea;">
            🛠️ Python + Streamlit + FastAPI<br>
            📈 Prophet + XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# ГЛАВНЫЙ ЗАГОЛОВОК
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>📊 Система прогнозирования продаж</h1>
    <p>ETL → Аналитика → Прогнозирование → Метрики</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# KPI ПАНЕЛЬ
# ---------------------------------------------------------
status = get_system_status()
raw_path = data_raw / 'sales_synth.csv'

if raw_path.exists():
    df_main = load_raw_data(str(raw_path))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="📦 Записей",
            value=f"{len(df_main):,}",
            delta="данные загружены"
        )

    with col2:
        st.metric(
            label="🏷️ SKU",
            value=df_main['sku_id'].nunique(),
            delta="уникальных товаров"
        )

    with col3:
        st.metric(
            label="🏪 Магазинов",
            value=df_main['store_id'].nunique(),
            delta="точек продаж"
        )

    with col4:
        date_range = (df_main['date'].max() - df_main['date'].min()).days
        st.metric(
            label="📅 Период",
            value=f"{date_range} дн.",
            delta="исторических данных"
        )

    with col5:
        avg_sales = df_main['units'].mean()
        st.metric(
            label="📈 Ср. продажи",
            value=f"{avg_sales:.1f}",
            delta="шт./день"
        )

st.divider()

# ---------------------------------------------------------
# ТАБЫ
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Данные",
    "📈 Прогноз",
    "📐 Аналитика",
    "⚙️ Модели",
    "🧮 Метрики"
])

# =====================================================================
# 📊 1. ДАННЫЕ
# =====================================================================
with tabs[0]:
    st.markdown('<div class="section-header">📊 Управление данными</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔧 Генерация данных")
        if st.button('🎲 Создать синтетические данные', use_container_width=True, type="primary"):
            with st.spinner('Генерация данных...'):
                success, error = run_command(
                    [sys.executable, '-m', 'src.etl.create_synthetic'],
                    "Генерация данных"
                )
                if success:
                    st.success('✅ Данные созданы!')
                    st.cache_data.clear()
                else:
                    st.error(f'❌ Ошибка: {error}')

    with col2:
        st.markdown("#### 🔄 ETL процесс")
        if st.button('⚙️ Запустить ETL', use_container_width=True):
            with st.spinner('Выполнение ETL...'):
                success, error = run_command(
                    [sys.executable, '-c', "from src.etl.prepare_dataset import main; main('data/raw/sales_synth.csv')"],
                    "ETL"
                )
                if success:
                    st.success('✅ ETL завершён!')
                    st.cache_data.clear()
                else:
                    st.error(f'❌ Ошибка: {error}')

    with col3:
        st.markdown("#### 📤 Загрузка файла")
        uploaded = st.file_uploader("CSV файл", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            user_path = data_raw / "sales_user.csv"
            df_u = pd.read_csv(uploaded)
            df_u.to_csv(user_path, index=False)
            st.success(f"✅ Файл сохранён!")

    st.divider()

    # Просмотр данных
    if raw_path.exists():
        df = load_raw_data(str(raw_path))

        st.markdown("#### 📋 Предпросмотр данных")

        # Фильтры
        col1, col2, col3 = st.columns(3)
        with col1:
            sku_filter = st.multiselect(
                "Фильтр по SKU",
                options=df['sku_id'].unique().tolist(),
                default=[]
            )
        with col2:
            store_filter = st.multiselect(
                "Фильтр по магазину",
                options=df['store_id'].unique().tolist(),
                default=[]
            )
        with col3:
            rows_to_show = st.slider("Количество строк", 10, 500, 100)

        # Применяем фильтры
        df_filtered = df.copy()
        if sku_filter:
            df_filtered = df_filtered[df_filtered['sku_id'].isin(sku_filter)]
        if store_filter:
            df_filtered = df_filtered[df_filtered['store_id'].isin(store_filter)]

        # Показываем таблицу с форматированием
        st.dataframe(
            df_filtered.head(rows_to_show).style.format({
                'units': '{:.0f}',
                'price': '{:.2f}' if 'price' in df_filtered.columns else '{}'
            }).background_gradient(subset=['units'], cmap='Blues'),
            use_container_width=True,
            height=400
        )

        # Статистика
        with st.expander("📊 Статистика по данным"):
            st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("👆 Нажмите «Создать синтетические данные» для начала работы")

# =====================================================================
# 📈 2. ПРОГНОЗ
# =====================================================================
with tabs[1]:
    st.markdown('<div class="section-header">📈 Прогнозирование продаж</div>', unsafe_allow_html=True)

    pred_path = data_proc / 'predictions.csv'

    if not raw_path.exists():
        st.warning("⚠️ Сначала загрузите данные во вкладке «Данные»")
    else:
        df_raw = load_raw_data(str(raw_path))

        # Панель управления
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])

        with col1:
            selected_sku = st.selectbox(
                "🏷️ Выберите SKU",
                options=df_raw['sku_id'].unique().tolist(),
                index=0
            )

        with col2:
            stores = ["Все"] + sorted(df_raw['store_id'].astype(str).unique().tolist())
            selected_store = st.selectbox("🏪 Магазин", stores)

        with col3:
            horizon = st.slider("📅 Горизонт (дни)", 7, 60, 14)

        with col4:
            models_selected = st.multiselect(
                "🤖 Модели",
                ["Prophet", "XGBoost", "Ensemble"],
                default=["Ensemble"]
            )

        # Кнопка прогноза
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Сделать прогноз", use_container_width=True, type="primary"):
                with st.spinner(f'Расчёт прогноза на {horizon} дней...'):
                    success, error = run_command(
                        [sys.executable, '-m', 'src.models.predict', '--horizon', str(horizon)],
                        "Прогноз"
                    )
                    if success:
                        st.success('✅ Прогноз готов!')
                        st.cache_data.clear()
                    else:
                        st.error(f'❌ Ошибка: {error}')

        st.divider()

        # Визуализация прогноза
        if pred_path.exists():
            df_pred = load_predictions_data(str(pred_path))

            # Фильтруем данные
            df_true = df_raw[df_raw['sku_id'] == selected_sku].copy()
            if selected_store != "Все":
                df_true = df_true[df_true['store_id'] == selected_store]
            df_true = df_true.sort_values('date').tail(90)

            # Фильтруем прогноз
            if 'sku_id' in df_pred.columns:
                df_p = df_pred[df_pred['sku_id'] == selected_sku].copy()
            else:
                df_p = df_pred.copy()

            if df_p.empty:
                st.warning("Нет данных для выбранного SKU. Перегенерируйте прогноз.")
            else:
                # Создаём интерактивный график Plotly
                fig = go.Figure()

                # Факт
                if not df_true.empty:
                    fig.add_trace(go.Scatter(
                        x=df_true['date'],
                        y=df_true['units'],
                        mode='lines+markers',
                        name='Факт',
                        line=dict(color=COLORS['fact'], width=2),
                        marker=dict(size=4)
                    ))

                # Модели
                model_cols = {"Prophet": "prophet", "XGBoost": "xgb", "Ensemble": "ensemble"}
                model_colors = {"Prophet": COLORS['prophet'], "XGBoost": COLORS['xgboost'], "Ensemble": COLORS['ensemble']}

                for model_name in models_selected:
                    col_name = model_cols.get(model_name)
                    if col_name and col_name in df_p.columns:
                        fig.add_trace(go.Scatter(
                            x=df_p['date'],
                            y=df_p[col_name],
                            mode='lines',
                            name=model_name,
                            line=dict(color=model_colors[model_name], width=3)
                        ))

                # Доверительный интервал для Prophet
                if 'p_low' in df_p.columns and 'p_high' in df_p.columns and 'Prophet' in models_selected:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([df_p['date'], df_p['date'][::-1]]),
                        y=pd.concat([df_p['p_high'], df_p['p_low'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(0, 212, 255, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Доверительный интервал'
                    ))

                fig.update_layout(
                    title=dict(
                        text=f"📈 Прогноз продаж: {selected_sku}",
                        font=dict(size=20)
                    ),
                    xaxis_title="Дата",
                    yaxis_title="Продажи, шт.",
                    template="plotly_white",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Таблица и скачивание
                col1, col2 = st.columns([3, 1])

                with col1:
                    with st.expander("📋 Таблица прогноза"):
                        st.dataframe(
                            df_p.style.format({
                                'prophet': '{:.1f}',
                                'xgb': '{:.1f}',
                                'ensemble': '{:.1f}'
                            }),
                            use_container_width=True
                        )

                with col2:
                    st.download_button(
                        "⬇️ Скачать CSV",
                        data=df_p.to_csv(index=False).encode('utf-8'),
                        file_name=f"forecast_{selected_sku}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("👆 Нажмите «Сделать прогноз» для генерации прогноза")

# =====================================================================
# 📐 3. АНАЛИТИКА
# =====================================================================
with tabs[2]:
    st.markdown('<div class="section-header">📐 Аналитика продаж</div>', unsafe_allow_html=True)

    if not raw_path.exists():
        st.warning("⚠️ Сначала загрузите данные")
    else:
        df = load_raw_data(str(raw_path))

        # Выбор SKU
        col1, col2 = st.columns([1, 3])
        with col1:
            sku_a = st.selectbox("🏷️ SKU для анализа", df['sku_id'].unique().tolist(), key="anal_sku")

        df_sku = df[df['sku_id'] == sku_a].sort_values('date')

        # Расчёт метрик
        df_sku['rolling_7'] = df_sku['units'].rolling(7, min_periods=1).mean()
        df_sku['rolling_30'] = df_sku['units'].rolling(30, min_periods=1).mean()

        # Метрики SKU
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Всего продаж", f"{df_sku['units'].sum():,.0f}")
        with col2:
            st.metric("📈 Среднее/день", f"{df_sku['units'].mean():.1f}")
        with col3:
            st.metric("📉 Мин.", f"{df_sku['units'].min():.0f}")
        with col4:
            st.metric("🔝 Макс.", f"{df_sku['units'].max():.0f}")

        st.divider()

        # График динамики
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "📈 Динамика продаж",
                "📊 Распределение продаж",
                "📅 Продажи по дням недели",
                "📆 Продажи по месяцам"
            ),
            specs=[[{"colspan": 2}, None], [{}, {}]]
        )

        # 1. Динамика
        fig.add_trace(
            go.Scatter(x=df_sku['date'], y=df_sku['units'], mode='lines',
                      name='Факт', line=dict(color='lightgray', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_sku['date'], y=df_sku['rolling_7'], mode='lines',
                      name='MA-7', line=dict(color=COLORS['prophet'], width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_sku['date'], y=df_sku['rolling_30'], mode='lines',
                      name='MA-30', line=dict(color=COLORS['xgboost'], width=2)),
            row=1, col=1
        )

        # 2. Гистограмма
        fig.add_trace(
            go.Histogram(x=df_sku['units'], nbinsx=30, name='Распределение',
                        marker_color=COLORS['primary']),
            row=2, col=1
        )

        # 3. По дням недели
        df_sku['dow'] = df_sku['date'].dt.dayofweek
        dow_sales = df_sku.groupby('dow')['units'].mean()
        dow_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        fig.add_trace(
            go.Bar(x=dow_names, y=dow_sales.values, name='По дням',
                  marker_color=COLORS['ensemble']),
            row=2, col=2
        )

        fig.update_layout(
            height=700,
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", y=1.1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Дополнительная статистика
        with st.expander("📊 Подробная статистика"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Описательная статистика**")
                st.dataframe(df_sku['units'].describe(), use_container_width=True)
            with col2:
                st.markdown("**Последние 10 записей**")
                st.dataframe(df_sku.tail(10), use_container_width=True)

# =====================================================================
# ⚙️ 4. МОДЕЛИ
# =====================================================================
with tabs[3]:
    st.markdown('<div class="section-header">⚙️ Обучение моделей</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>🔮 Prophet</h3>
            <p>Модель временных рядов от Meta</p>
            <p style="font-size: 0.9rem; color: #666;">
                Автоматическое обнаружение сезонности и трендов
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button('🚀 Обучить Prophet', use_container_width=True, type="primary"):
            with st.spinner('Обучение Prophet...'):
                success, error = run_command(
                    [sys.executable, '-m', 'src.models.train_prophet'],
                    "Prophet"
                )
                if success:
                    st.success('✅ Prophet обучен!')
                    st.cache_data.clear()
                    st.balloons()
                else:
                    st.error(f'❌ Ошибка: {error}')

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b22, #ffa50222);
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>🌲 XGBoost</h3>
            <p>Градиентный бустинг</p>
            <p style="font-size: 0.9rem; color: #666;">
                Машинное обучение с признаками
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button('🚀 Обучить XGBoost', use_container_width=True, type="primary"):
            with st.spinner('Обучение XGBoost...'):
                success, error = run_command(
                    [sys.executable, '-m', 'src.models.train_xgboost'],
                    "XGBoost"
                )
                if success:
                    st.success('✅ XGBoost обучен!')
                    st.cache_data.clear()
                    st.balloons()
                else:
                    st.error(f'❌ Ошибка: {error}')

    st.divider()

    # Статус моделей
    st.markdown("### 📋 Статус моделей")
    status = get_system_status()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if status["prophet"]:
            st.success("Prophet: Обучен ✅")
        else:
            st.warning("Prophet: Не обучен ⚠️")

    with col2:
        if status["xgboost"]:
            st.success("XGBoost: Обучен ✅")
        else:
            st.warning("XGBoost: Не обучен ⚠️")

    with col3:
        if status["predictions"]:
            st.success("Прогнозы: Есть ✅")
        else:
            st.info("Прогнозы: Нет")

    with col4:
        if status["metrics"]:
            st.success("Метрики: Есть ✅")
        else:
            st.info("Метрики: Нет")

# =====================================================================
# 🧮 5. МЕТРИКИ
# =====================================================================
with tabs[4]:
    st.markdown('<div class="section-header">🧮 Метрики качества</div>', unsafe_allow_html=True)

    metrics_path = data_proc / 'metrics.csv'

    # Кнопка пересчёта
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Пересчитать метрики", use_container_width=True, type="primary"):
            with st.spinner('Расчёт метрик...'):
                success, error = run_command(
                    [sys.executable, '-m', 'src.models.evaluate', '--horizon', '14'],
                    "Метрики"
                )
                if success:
                    st.success('✅ Метрики пересчитаны!')
                    st.cache_data.clear()
                else:
                    st.error(f'❌ Ошибка: {error}')

    st.divider()

    if metrics_path.exists():
        met = load_metrics_data(str(metrics_path))

        # Средние метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_prophet = met['mape_prophet'].mean()
            st.metric("🔮 Prophet MAPE", f"{avg_prophet:.1f}%")

        with col2:
            avg_xgb = met['mape_xgboost'].mean()
            st.metric("🌲 XGBoost MAPE", f"{avg_xgb:.1f}%")

        with col3:
            avg_ens = met['mape_ens'].mean()
            st.metric("🎯 Ensemble MAPE", f"{avg_ens:.1f}%")

        with col4:
            best_count = met['best_model'].value_counts()
            best_model = best_count.index[0] if len(best_count) > 0 else "N/A"
            st.metric("🏆 Лучшая модель", best_model.upper())

        st.divider()

        # Графики сравнения
        col1, col2 = st.columns(2)

        with col1:
            # Сравнение моделей
            fig_compare = go.Figure()

            models_data = ['mape_prophet', 'mape_xgboost', 'mape_naive', 'mape_ens']
            models_names = ['Prophet', 'XGBoost', 'Naive', 'Ensemble']
            colors = [COLORS['prophet'], COLORS['xgboost'], '#888888', COLORS['ensemble']]

            for col, name, color in zip(models_data, models_names, colors):
                fig_compare.add_trace(go.Box(
                    y=met[col],
                    name=name,
                    marker_color=color
                ))

            fig_compare.update_layout(
                title="📊 Сравнение моделей (MAPE)",
                yaxis_title="MAPE, %",
                template="plotly_white",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_compare, use_container_width=True)

        with col2:
            # Pie chart побед
            wins = met['best_model'].value_counts()

            fig_pie = go.Figure(data=[go.Pie(
                labels=wins.index.str.upper(),
                values=wins.values,
                hole=0.4,
                marker_colors=[COLORS['prophet'], COLORS['xgboost'], COLORS['ensemble'], '#888888'][:len(wins)]
            )])

            fig_pie.update_layout(
                title="🏆 Доля побед моделей",
                template="plotly_white",
                height=400
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        # Детальная таблица
        with st.expander("📋 Полная таблица метрик"):
            st.dataframe(
                met.style.format({
                    'mape_prophet': '{:.1f}%',
                    'mape_xgboost': '{:.1f}%',
                    'mape_naive': '{:.1f}%',
                    'mape_ens': '{:.1f}%'
                }).background_gradient(subset=['mape_ens'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )

        # Паспорт модели
        st.divider()
        st.markdown("### 📋 Паспорт модели по SKU")

        sel_sku = st.selectbox("Выберите SKU", met['sku_id'].unique().tolist(), key="passport_sku")
        row = met[met['sku_id'] == sel_sku].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea22, #764ba222);
                        padding: 1.5rem; border-radius: 15px;">
                <h4>🏷️ {sel_sku}</h4>
                <p><strong>Лучшая модель:</strong> {row['best_model'].upper()}</p>
                <hr>
                <p>Prophet: <strong>{row['mape_prophet']:.1f}%</strong></p>
                <p>XGBoost: <strong>{row['mape_xgboost']:.1f}%</strong></p>
                <p>Naive: <strong>{row['mape_naive']:.1f}%</strong></p>
                <p>Ensemble: <strong>{row['mape_ens']:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig_sku = go.Figure(data=[
                go.Bar(
                    x=['Prophet', 'XGBoost', 'Naive', 'Ensemble'],
                    y=[row['mape_prophet'], row['mape_xgboost'], row['mape_naive'], row['mape_ens']],
                    marker_color=[COLORS['prophet'], COLORS['xgboost'], '#888888', COLORS['ensemble']]
                )
            ])

            fig_sku.update_layout(
                title=f"MAPE для {sel_sku}",
                yaxis_title="MAPE, %",
                template="plotly_white",
                height=300
            )

            st.plotly_chart(fig_sku, use_container_width=True)

        # Скачивание
        st.download_button(
            "⬇️ Скачать metrics.csv",
            data=metrics_path.read_bytes(),
            file_name="metrics.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Нажмите «Пересчитать метрики» для расчёта качества моделей")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>📊 Forecastly v1.0 | Дипломный проект | 2025</p>
    <p style="font-size: 0.8rem;">Вульферт Н.Е. | Группа 122 ИСП | Новосибирский политехнический колледж</p>
</div>
""", unsafe_allow_html=True)
