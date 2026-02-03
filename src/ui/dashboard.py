"""
Streamlit дашборд для системы прогнозирования продаж Forecastly.

Разделы:
- Данные: загрузка CSV/XLSX, DQ отчёт, история загрузок
- Прогноз: выбор SKU, горизонт 1-30, модели (Prophet/XGBoost/LightGBM/Ensemble)
- Аналитика: KPI, графики, сезонность
- Модели: обучение Prophet/XGBoost/LightGBM, статус
- Метрики: MAE/RMSE/MAPE/sMAPE, сравнение моделей, best model per SKU
- Админ: пользователи/роли (заглушка для UI, работает через API)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from io import BytesIO
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

def check_role(required_roles: list[str]) -> bool:
    """Проверяет, есть ли у текущего пользователя нужная роль."""
    return st.session_state.get("user_role") in required_roles

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #333; }
    .kpi-label { font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, transparent 100%);
        padding: 0.8rem 1.5rem; border-radius: 8px;
        color: white; font-weight: 600; margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        border-radius: 25px; padding: 0.5rem 2rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 20px; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 2rem; }
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

COLORS = {
    'primary': '#667eea', 'secondary': '#764ba2', 'success': '#38ef7d',
    'warning': '#f5576c', 'prophet': '#00d4ff', 'xgboost': '#ff6b6b',
    'lightgbm': '#ffa502', 'ensemble': '#4ecdc4', 'fact': '#2d3436',
    'naive': '#888888', 'background': '#f8f9fa'
}


# ---------------------------------------------------------
# Кэшированные функции
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_system_status():
    return {
        "raw_data": (data_raw / 'sales_synth.csv').exists() or any(data_raw.glob("*.csv")),
        "predictions": (data_proc / 'predictions.csv').exists(),
        "metrics": (data_proc / 'metrics.csv').exists(),
        "prophet": (data_models / 'prophet_model.pkl').exists(),
        "xgboost": (data_models / 'xgboost_model.pkl').exists(),
        "lightgbm": (data_models / 'lightgbm_model.pkl').exists(),
    }


@st.cache_data
def load_raw_data(file_path: str = 'data/raw/sales_synth.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df


@st.cache_data
def load_predictions_data(file_path: str = 'data/processed/predictions.csv') -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=['date'])


@st.cache_data
def load_metrics_data(file_path: str = 'data/processed/metrics.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)


def run_command(cmd: list, description: str) -> tuple:
    """Выполняет команду с timeout для безопасности."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Команда превысила лимит времени выполнения (300 секунд)"


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в XLSX bytes для скачивания."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()


def compute_dq_report(df: pd.DataFrame) -> dict:
    """Формирует отчёт о качестве данных (DQ report)."""
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "sku_count": int(df['sku_id'].nunique()) if 'sku_id' in df.columns else 0,
        "date_range": None,
        "missing_pct": {},
        "duplicates": int(df.duplicated().sum()),
        "outliers": {},
    }

    if 'date' in df.columns:
        report["date_range"] = {
            "min": str(df['date'].min()),
            "max": str(df['date'].max()),
            "days": int((df['date'].max() - df['date'].min()).days) if len(df) > 1 else 0,
        }

    for col in df.columns:
        pct = float(df[col].isna().mean() * 100)
        if pct > 0:
            report["missing_pct"][col] = round(pct, 1)

    # Выбросы по IQR для числовых столбцов
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if outliers > 0:
                report["outliers"][col] = outliers

    return report


# ---------------------------------------------------------
# АВТОРИЗАЦИЯ
# ---------------------------------------------------------
API_URL = "http://localhost:8000"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_email = None
    st.session_state.token = None

def do_login(email: str, password: str) -> bool:
    """Authenticate via API and store token."""
    import requests
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/auth/login",
            json={"email": email, "password": password},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.authenticated = True
            st.session_state.token = data.get("access_token", "")
            st.session_state.user_email = email
            st.session_state.user_role = data.get("role", "viewer")
            return True
    except Exception:
        pass
    return False

def do_demo_login(role: str):
    """Demo login without API (for development/demo mode)."""
    st.session_state.authenticated = True
    st.session_state.user_email = f"demo_{role}@forecastly.local"
    st.session_state.user_role = role
    st.session_state.token = "demo"

if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Forecastly</h1>
        <p>Система прогнозирования продаж</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("### Вход в систему")

        login_tab, demo_tab = st.tabs(["Авторизация", "Демо-режим"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="user@example.com")
                password = st.text_input("Пароль", type="password")
                submitted = st.form_submit_button("Войти", use_container_width=True)

                if submitted and email and password:
                    if do_login(email, password):
                        st.rerun()
                    else:
                        st.error("Неверный email или пароль")

        with demo_tab:
            st.info("Демо-режим позволяет войти без подключения к API")
            demo_role = st.selectbox("Выберите роль", ["admin", "analyst", "viewer"])
            if st.button("Войти в демо-режиме", use_container_width=True):
                do_demo_login(demo_role)
                st.rerun()

    st.stop()

# ---------------------------------------------------------
# RBAC: ограничение табов по ролям
# ---------------------------------------------------------
ROLE_TABS = {
    "admin": ["Данные", "Прогноз", "Аналитика", "Модели", "Метрики", "Админ"],
    "analyst": ["Данные", "Прогноз", "Аналитика", "Модели", "Метрики"],
    "viewer": ["Прогноз", "Аналитика", "Метрики"],
}
allowed_tabs = ROLE_TABS.get(st.session_state.user_role, ROLE_TABS["viewer"])

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

    # Информация о пользователе
    role_labels = {"admin": "Администратор", "analyst": "Аналитик", "viewer": "Наблюдатель"}
    user_role = st.session_state.get("user_role", "viewer")
    user_email = st.session_state.get("user_email", "")
    st.markdown(
        f"**{user_email}**  \n"
        f"Роль: `{role_labels.get(user_role, user_role)}`"
    )
    if st.button("Выйти", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.user_email = None
        st.session_state.token = None
        st.rerun()

    st.divider()

    status = get_system_status()
    st.markdown("### Статус системы")

    col1, col2 = st.columns(2)
    with col1:
        st.success("Данные") if status["raw_data"] else st.error("Данные")
        st.success("Прогноз") if status["predictions"] else st.warning("Прогноз")
    with col2:
        st.success("Prophet") if status["prophet"] else st.warning("Prophet")
        st.success("XGBoost") if status["xgboost"] else st.warning("XGBoost")

    if status["lightgbm"]:
        st.success("LightGBM")
    else:
        st.warning("LightGBM")

    st.divider()

    if st.button("Обновить статус", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### О проекте")
    st.markdown("""
    **Автор:** Вульферт Н.Е.
    **Группа:** 122 ИСП
    **Год:** 2026
    """)
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
        <p style="margin: 0; font-size: 0.8rem; color: #667eea;">
            Python + Streamlit + FastAPI<br>
            Prophet + XGBoost + LightGBM
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
        st.metric(label="Записей", value=f"{len(df_main):,}", delta="данные загружены")
    with col2:
        st.metric(label="SKU", value=df_main['sku_id'].nunique(), delta="уникальных товаров")
    with col3:
        st.metric(label="Магазинов", value=df_main['store_id'].nunique(), delta="точек продаж")
    with col4:
        date_range = (df_main['date'].max() - df_main['date'].min()).days
        st.metric(label="Период", value=f"{date_range} дн.", delta="исторических данных")
    with col5:
        avg_sales = df_main['units'].mean()
        st.metric(label="Ср. продажи", value=f"{avg_sales:.1f}", delta="шт./день")

st.divider()

# ---------------------------------------------------------
# ТАБЫ
# ---------------------------------------------------------
all_tab_defs = [
    ("📊 Данные", "Данные"),
    ("📈 Прогноз", "Прогноз"),
    ("📐 Аналитика", "Аналитика"),
    ("⚙️ Модели", "Модели"),
    ("🧮 Метрики", "Метрики"),
    ("🔧 Админ", "Админ"),
]
visible_tab_defs = [(label, key) for label, key in all_tab_defs if key in allowed_tabs]
tabs = st.tabs([label for label, _ in visible_tab_defs])
tab_keys = [key for _, key in visible_tab_defs]

def get_tab(name):
    """Get tab by key name, returns None if not visible for current role."""
    if name in tab_keys:
        return tabs[tab_keys.index(name)]
    return None

# =====================================================================
# 📊 1. ДАННЫЕ (с DQ отчётом и XLSX поддержкой)
# =====================================================================
with tabs[tab_keys.index("Данные")] if "Данные" in tab_keys else st.container():
    st.markdown('<div class="section-header">📊 Управление данными</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    can_edit_data = check_role(["admin", "analyst"])

    if can_edit_data:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Генерация данных")
            if st.button('Создать синтетические данные', use_container_width=True, type="primary"):
                with st.spinner('Генерация данных...'):
                    success, error = run_command(
                        [sys.executable, '-m', 'src.etl.create_synthetic'], "Генерация"
                    )
                    if success:
                        st.success('Данные созданы!')
                        st.cache_data.clear()
                    else:
                        st.error(f'Ошибка: {error}')

        with col2:
            st.markdown("#### ETL процесс")
            if st.button('Запустить ETL', use_container_width=True):
                with st.spinner('Выполнение ETL...'):
                    success, error = run_command(
                        [sys.executable, '-c',
                         "from src.etl.prepare_dataset import main; main('data/raw/sales_synth.csv')"],
                        "ETL"
                    )
                    if success:
                        st.success('ETL завершён!')
                        st.cache_data.clear()
                    else:
                        st.error(f'Ошибка: {error}')

        with col3:
            st.markdown("#### Загрузка файла")
            uploaded = st.file_uploader("CSV/XLSX файл", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.xlsx'):
                    df_u = pd.read_excel(uploaded, engine='openpyxl')
                else:
                    df_u = pd.read_csv(uploaded)

                # Нормализация колонок
                df_u.columns = [c.strip().lower() for c in df_u.columns]
                col_map = {"qty": "units", "quantity": "units", "sales": "units"}
                df_u.rename(columns=col_map, inplace=True)

                user_path = data_raw / "sales_user.csv"
                df_u.to_csv(user_path, index=False)
                st.success(f"Файл сохранён! ({len(df_u)} строк)")
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
    else:
        st.info("Загрузка и обработка данных доступна для ролей admin и analyst.")

    st.divider()

    # DQ отчёт
    if raw_path.exists():
        df = load_raw_data(str(raw_path))

        with st.expander("📋 Отчёт о качестве данных (DQ Report)", expanded=False):
            dq = compute_dq_report(df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Строк", f"{dq['rows']:,}")
            with col2:
                st.metric("SKU", dq['sku_count'])
            with col3:
                if dq['date_range']:
                    st.metric("Период (дни)", dq['date_range']['days'])
            with col4:
                st.metric("Дубликатов", dq['duplicates'])

            if dq['missing_pct']:
                st.markdown("**Пропуски по столбцам:**")
                miss_df = pd.DataFrame(
                    [{"Столбец": k, "Пропуски, %": v} for k, v in dq['missing_pct'].items()]
                )
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            else:
                st.success("Пропусков нет")

            if dq['outliers']:
                st.markdown("**Выбросы (IQR метод):**")
                out_df = pd.DataFrame(
                    [{"Столбец": k, "Выбросов": v} for k, v in dq['outliers'].items()]
                )
                st.dataframe(out_df, use_container_width=True, hide_index=True)
            else:
                st.success("Выбросов не обнаружено")

        # Просмотр данных
        st.markdown("#### Предпросмотр данных")
        col1, col2, col3 = st.columns(3)
        with col1:
            sku_filter = st.multiselect("Фильтр по SKU", df['sku_id'].unique().tolist(), default=[])
        with col2:
            store_filter = st.multiselect("Фильтр по магазину", df['store_id'].unique().tolist(), default=[])
        with col3:
            rows_to_show = st.slider("Количество строк", 10, 500, 100)

        df_filtered = df.copy()
        if sku_filter:
            df_filtered = df_filtered[df_filtered['sku_id'].isin(sku_filter)]
        if store_filter:
            df_filtered = df_filtered[df_filtered['store_id'].isin(store_filter)]

        st.dataframe(df_filtered.head(rows_to_show), use_container_width=True, height=400)

        with st.expander("Статистика по данным"):
            st.dataframe(df.describe(), use_container_width=True)

        # История загрузок
        with st.expander("История загрузок"):
            uploads = sorted(data_raw.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            for f in uploads[:10]:
                mod = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size_kb = f.stat().st_size / 1024
                st.text(f"{f.name}  |  {size_kb:.0f} KB  |  {mod}")
    else:
        st.info("Нажмите «Создать синтетические данные» для начала работы")

# =====================================================================
# 📈 2. ПРОГНОЗ
# =====================================================================
with tabs[tab_keys.index("Прогноз")] if "Прогноз" in tab_keys else st.container():
    st.markdown('<div class="section-header">📈 Прогнозирование продаж</div>', unsafe_allow_html=True)

    pred_path = data_proc / 'predictions.csv'

    if not raw_path.exists():
        st.warning("Сначала загрузите данные во вкладке «Данные»")
    else:
        df_raw = load_raw_data(str(raw_path))

        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])

        with col1:
            selected_sku = st.selectbox("Выберите SKU", df_raw['sku_id'].unique().tolist(), index=0)
        with col2:
            stores = ["Все"] + sorted(df_raw['store_id'].astype(str).unique().tolist())
            selected_store = st.selectbox("Магазин", stores)
        with col3:
            horizon = st.slider("Горизонт (дни)", 1, 30, 14)
        with col4:
            models_selected = st.multiselect(
                "Модели",
                ["Prophet", "XGBoost", "LightGBM", "Ensemble"],
                default=["Ensemble"]
            )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Сделать прогноз", use_container_width=True, type="primary"):
                with st.spinner(f'Расчёт прогноза на {horizon} дней...'):
                    success, error = run_command(
                        [sys.executable, '-m', 'src.models.predict', '--horizon', str(horizon)],
                        "Прогноз"
                    )
                    if success:
                        st.success('Прогноз готов!')
                        st.cache_data.clear()
                    else:
                        st.error(f'Ошибка: {error}')

        st.divider()

        if pred_path.exists():
            df_pred = load_predictions_data(str(pred_path))

            df_true = df_raw[df_raw['sku_id'] == selected_sku].copy()
            if selected_store != "Все":
                df_true = df_true[df_true['store_id'] == selected_store]
            df_true = df_true.sort_values('date').tail(90)

            if 'sku_id' in df_pred.columns:
                df_p = df_pred[df_pred['sku_id'] == selected_sku].copy()
            else:
                df_p = df_pred.copy()

            if df_p.empty:
                st.warning("Нет данных для выбранного SKU.")
            else:
                fig = go.Figure()

                if not df_true.empty:
                    fig.add_trace(go.Scatter(
                        x=df_true['date'], y=df_true['units'],
                        mode='lines+markers', name='Факт',
                        line=dict(color=COLORS['fact'], width=2), marker=dict(size=4)
                    ))

                model_cols = {
                    "Prophet": "prophet", "XGBoost": "xgb",
                    "LightGBM": "lgbm", "Ensemble": "ensemble"
                }
                model_colors = {
                    "Prophet": COLORS['prophet'], "XGBoost": COLORS['xgboost'],
                    "LightGBM": COLORS['lightgbm'], "Ensemble": COLORS['ensemble']
                }

                for model_name in models_selected:
                    col_name = model_cols.get(model_name)
                    if col_name and col_name in df_p.columns:
                        fig.add_trace(go.Scatter(
                            x=df_p['date'], y=df_p[col_name],
                            mode='lines', name=model_name,
                            line=dict(color=model_colors[model_name], width=3)
                        ))

                # Доверительный интервал Prophet
                if 'p_low' in df_p.columns and 'p_high' in df_p.columns and 'Prophet' in models_selected:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([df_p['date'], df_p['date'][::-1]]),
                        y=pd.concat([df_p['p_high'], df_p['p_low'][::-1]]),
                        fill='toself', fillcolor='rgba(0, 212, 255, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'), name='Доверительный интервал'
                    ))

                fig.update_layout(
                    title=dict(text=f"Прогноз продаж: {selected_sku}", font=dict(size=20)),
                    xaxis_title="Дата", yaxis_title="Продажи, шт.",
                    template="plotly_white", height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Таблица и скачивание
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    with st.expander("Таблица прогноза"):
                        fmt = {}
                        for c in ['prophet', 'xgb', 'lgbm', 'ensemble']:
                            if c in df_p.columns:
                                fmt[c] = '{:.1f}'
                        st.dataframe(df_p.style.format(fmt), use_container_width=True)

                with col2:
                    st.download_button(
                        "Скачать CSV", data=df_p.to_csv(index=False).encode('utf-8'),
                        file_name=f"forecast_{selected_sku}.csv", mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    st.download_button(
                        "Скачать XLSX", data=to_excel_bytes(df_p),
                        file_name=f"forecast_{selected_sku}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        else:
            st.info("Нажмите «Сделать прогноз» для генерации")

# =====================================================================
# 📐 3. АНАЛИТИКА
# =====================================================================
with tabs[tab_keys.index("Аналитика")] if "Аналитика" in tab_keys else st.container():
    st.markdown('<div class="section-header">📐 Аналитика продаж</div>', unsafe_allow_html=True)

    if not raw_path.exists():
        st.warning("Сначала загрузите данные")
    else:
        df = load_raw_data(str(raw_path))

        col1, col2 = st.columns([1, 3])
        with col1:
            sku_a = st.selectbox("SKU для анализа", df['sku_id'].unique().tolist(), key="anal_sku")

        df_sku = df[df['sku_id'] == sku_a].sort_values('date')
        df_sku['rolling_7'] = df_sku['units'].rolling(7, min_periods=1).mean()
        df_sku['rolling_30'] = df_sku['units'].rolling(30, min_periods=1).mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего продаж", f"{df_sku['units'].sum():,.0f}")
        with col2:
            st.metric("Среднее/день", f"{df_sku['units'].mean():.1f}")
        with col3:
            st.metric("Мин.", f"{df_sku['units'].min():.0f}")
        with col4:
            st.metric("Макс.", f"{df_sku['units'].max():.0f}")

        st.divider()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Динамика продаж", "Распределение", "По дням недели", "По месяцам"),
            specs=[[{"colspan": 2}, None], [{}, {}]]
        )

        fig.add_trace(go.Scatter(x=df_sku['date'], y=df_sku['units'], mode='lines',
                                  name='Факт', line=dict(color='lightgray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_sku['date'], y=df_sku['rolling_7'], mode='lines',
                                  name='MA-7', line=dict(color=COLORS['prophet'], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_sku['date'], y=df_sku['rolling_30'], mode='lines',
                                  name='MA-30', line=dict(color=COLORS['xgboost'], width=2)), row=1, col=1)

        fig.add_trace(go.Histogram(x=df_sku['units'], nbinsx=30, name='Распределение',
                                    marker_color=COLORS['primary']), row=2, col=1)

        df_sku['dow'] = df_sku['date'].dt.dayofweek
        dow_sales = df_sku.groupby('dow')['units'].mean()
        dow_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        fig.add_trace(go.Bar(x=dow_names, y=dow_sales.values, name='По дням',
                              marker_color=COLORS['ensemble']), row=2, col=2)

        fig.update_layout(height=700, template="plotly_white", showlegend=True,
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Подробная статистика"):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_sku['units'].describe(), use_container_width=True)
            with col2:
                st.dataframe(df_sku.tail(10), use_container_width=True)

# =====================================================================
# ⚙️ 4. МОДЕЛИ (с LightGBM)
# =====================================================================
with tabs[tab_keys.index("Модели")] if "Модели" in tab_keys else st.container():
    st.markdown('<div class="section-header">⚙️ Обучение моделей</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>Prophet</h3>
            <p>Временные ряды (Meta)</p>
            <p style="font-size: 0.9rem; color: #666;">Сезонность и тренды</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button('Обучить Prophet', use_container_width=True, type="primary"):
            with st.spinner('Обучение Prophet...'):
                success, error = run_command([sys.executable, '-m', 'src.models.train_prophet'], "Prophet")
                if success:
                    st.success('Prophet обучен!')
                    st.cache_data.clear()
                else:
                    st.error(f'Ошибка: {error}')

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b22, #ffa50222);
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>XGBoost</h3>
            <p>Градиентный бустинг</p>
            <p style="font-size: 0.9rem; color: #666;">ML с признаками</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button('Обучить XGBoost', use_container_width=True, type="primary"):
            with st.spinner('Обучение XGBoost...'):
                success, error = run_command([sys.executable, '-m', 'src.models.train_xgboost'], "XGBoost")
                if success:
                    st.success('XGBoost обучен!')
                    st.cache_data.clear()
                else:
                    st.error(f'Ошибка: {error}')

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffa50222, #38ef7d22);
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>LightGBM</h3>
            <p>Быстрый градиентный бустинг</p>
            <p style="font-size: 0.9rem; color: #666;">Быстрее XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button('Обучить LightGBM', use_container_width=True, type="primary"):
            with st.spinner('Обучение LightGBM...'):
                success, error = run_command([sys.executable, '-m', 'src.models.train_lightgbm'], "LightGBM")
                if success:
                    st.success('LightGBM обучен!')
                    st.cache_data.clear()
                else:
                    st.error(f'Ошибка: {error}')

    st.divider()
    st.markdown("### Статус моделей")
    status = get_system_status()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.success("Prophet: Обучен") if status["prophet"] else st.warning("Prophet: Не обучен")
    with col2:
        st.success("XGBoost: Обучен") if status["xgboost"] else st.warning("XGBoost: Не обучен")
    with col3:
        st.success("LightGBM: Обучен") if status["lightgbm"] else st.warning("LightGBM: Не обучен")
    with col4:
        st.success("Прогнозы: Есть") if status["predictions"] else st.info("Прогнозы: Нет")
    with col5:
        st.success("Метрики: Есть") if status["metrics"] else st.info("Метрики: Нет")

# =====================================================================
# 🧮 5. МЕТРИКИ (с MAE/RMSE/sMAPE и LightGBM)
# =====================================================================
with tabs[tab_keys.index("Метрики")] if "Метрики" in tab_keys else st.container():
    st.markdown('<div class="section-header">🧮 Метрики качества и сравнение моделей</div>', unsafe_allow_html=True)

    metrics_path = data_proc / 'metrics.csv'

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Пересчитать метрики", use_container_width=True, type="primary"):
            with st.spinner('Расчёт метрик (backtesting)...'):
                success, error = run_command(
                    [sys.executable, '-m', 'src.models.evaluate', '--horizon', '14', '--folds', '3'],
                    "Метрики"
                )
                if success:
                    st.success('Метрики пересчитаны!')
                    st.cache_data.clear()
                else:
                    st.error(f'Ошибка: {error}')

    st.divider()

    if metrics_path.exists():
        met = load_metrics_data(str(metrics_path))

        # Средние метрики MAPE
        model_names = []
        model_mapes = []
        model_colors_list = []

        for mname, col_name, color in [
            ('Prophet', 'mape_prophet', COLORS['prophet']),
            ('XGBoost', 'mape_xgboost', COLORS['xgboost']),
            ('LightGBM', 'mape_lightgbm', COLORS['lightgbm']),
            ('Naive', 'mape_naive', COLORS['naive']),
            ('Ensemble', 'mape_ens', COLORS['ensemble']),
        ]:
            if col_name in met.columns:
                model_names.append(mname)
                model_mapes.append(met[col_name].mean())
                model_colors_list.append(color)

        cols = st.columns(len(model_names) + 1)
        for i, (name, mape_val) in enumerate(zip(model_names, model_mapes)):
            with cols[i]:
                st.metric(f"{name} MAPE", f"{mape_val:.1f}%")

        if 'best_model' in met.columns:
            with cols[-1]:
                best_count = met['best_model'].value_counts()
                best_model = best_count.index[0] if len(best_count) > 0 else "N/A"
                st.metric("Лучшая модель", best_model.upper())

        st.divider()

        # Графики сравнения
        col1, col2 = st.columns(2)

        with col1:
            fig_compare = go.Figure()
            for name, color in zip(model_names, model_colors_list):
                col_name = f"mape_{name.lower()}" if name != "Ensemble" else "mape_ens"
                if col_name in met.columns:
                    fig_compare.add_trace(go.Box(y=met[col_name], name=name, marker_color=color))

            fig_compare.update_layout(
                title="Сравнение моделей (MAPE)", yaxis_title="MAPE, %",
                template="plotly_white", height=400, showlegend=False
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with col2:
            if 'best_model' in met.columns:
                wins = met['best_model'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=wins.index.str.upper(), values=wins.values, hole=0.4,
                    marker_colors=[COLORS.get(n.lower(), '#888') for n in wins.index]
                )])
                fig_pie.update_layout(title="Доля побед моделей", template="plotly_white", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

        # MAE / RMSE сравнение (если есть)
        mae_cols = [c for c in met.columns if c.startswith('mae_')]
        rmse_cols = [c for c in met.columns if c.startswith('rmse_')]

        if mae_cols or rmse_cols:
            st.markdown("### MAE и RMSE по моделям")
            col1, col2 = st.columns(2)

            with col1:
                if mae_cols:
                    fig_mae = go.Figure()
                    for col_name in mae_cols:
                        mname = col_name.replace("mae_", "").upper()
                        fig_mae.add_trace(go.Box(y=met[col_name], name=mname))
                    fig_mae.update_layout(title="MAE по моделям", template="plotly_white", height=350)
                    st.plotly_chart(fig_mae, use_container_width=True)

            with col2:
                if rmse_cols:
                    fig_rmse = go.Figure()
                    for col_name in rmse_cols:
                        mname = col_name.replace("rmse_", "").upper()
                        fig_rmse.add_trace(go.Box(y=met[col_name], name=mname))
                    fig_rmse.update_layout(title="RMSE по моделям", template="plotly_white", height=350)
                    st.plotly_chart(fig_rmse, use_container_width=True)

        # Детальная таблица
        with st.expander("Полная таблица метрик"):
            fmt = {}
            for c in met.columns:
                if c.startswith(('mape_', 'mae_', 'rmse_', 'smape_')):
                    fmt[c] = '{:.1f}'
            st.dataframe(met.style.format(fmt), use_container_width=True, height=400)

        # Паспорт модели по SKU
        st.divider()
        st.markdown("### Паспорт модели по SKU")

        sel_sku = st.selectbox("Выберите SKU", met['sku_id'].unique().tolist(), key="passport_sku")
        row = met[met['sku_id'] == sel_sku].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            info_lines = [f"**Лучшая модель:** {row.get('best_model', 'N/A').upper()}"]
            for prefix in ['prophet', 'xgboost', 'lightgbm', 'naive', 'ens']:
                mape_val = row.get(f'mape_{prefix}', np.nan)
                mae_val = row.get(f'mae_{prefix}', np.nan)
                rmse_val = row.get(f'rmse_{prefix}', np.nan)
                if not np.isnan(mape_val):
                    info_lines.append(
                        f"**{prefix.upper()}**: MAPE={mape_val:.1f}%  MAE={mae_val:.1f}  RMSE={rmse_val:.1f}"
                    )
            st.markdown(f"#### {sel_sku}")
            st.markdown("\n\n".join(info_lines))

        with col2:
            bar_models = []
            bar_values = []
            bar_colors = []
            for prefix, color in [
                ('prophet', COLORS['prophet']), ('xgboost', COLORS['xgboost']),
                ('lightgbm', COLORS['lightgbm']), ('naive', COLORS['naive']),
                ('ens', COLORS['ensemble'])
            ]:
                val = row.get(f'mape_{prefix}', np.nan)
                if not np.isnan(val):
                    bar_models.append(prefix.upper())
                    bar_values.append(val)
                    bar_colors.append(color)

            fig_sku = go.Figure(data=[go.Bar(x=bar_models, y=bar_values, marker_color=bar_colors)])
            fig_sku.update_layout(title=f"MAPE для {sel_sku}", yaxis_title="MAPE, %",
                                   template="plotly_white", height=300)
            st.plotly_chart(fig_sku, use_container_width=True)

        # Скачивание метрик
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Скачать metrics.csv", data=metrics_path.read_bytes(),
                                file_name="metrics.csv", mime="text/csv")
        with col2:
            st.download_button("Скачать metrics.xlsx", data=to_excel_bytes(met),
                                file_name="metrics.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Нажмите «Пересчитать метрики» для расчёта (backtesting)")

# =====================================================================
# 🔧 6. АДМИН-ПАНЕЛЬ
# =====================================================================
with tabs[tab_keys.index("Админ")] if "Админ" in tab_keys else st.container():
    st.markdown('<div class="section-header">🔧 Администрирование</div>', unsafe_allow_html=True)

    admin_tabs = st.tabs(["Система", "Пользователи", "Аудит", "Настройки"])

    # --- Система ---
    with admin_tabs[0]:
        st.markdown("### Системная информация")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Версии компонентов:**")
            st.text("Forecastly API: v1.4.0")
            st.text(f"Python: {sys.version.split()[0]}")
            try:
                import streamlit as _st_ver
                st.text(f"Streamlit: {_st_ver.__version__}")
            except Exception:
                pass

        with col2:
            st.markdown("**Пути данных:**")
            st.text(f"Raw: {data_raw}")
            st.text(f"Processed: {data_proc}")
            st.text(f"Models: {data_models}")

        with col3:
            st.markdown("**Размеры данных:**")
            for p in [data_raw, data_proc, data_models]:
                if p.exists():
                    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    st.text(f"{p.name}: {total / 1024 / 1024:.1f} MB")

        st.divider()

        st.markdown("### API Endpoints")
        st.markdown("""
        | Endpoint | Метод | Описание |
        |----------|-------|----------|
        | `/api/v1/admin/users` | GET/POST | Управление пользователями |
        | `/api/v1/admin/users/{id}` | PATCH | Обновление пользователя |
        | `/api/v1/admin/roles` | GET | Список ролей |
        | `/api/v1/admin/audit` | GET | Журнал аудита |
        | `/api/v1/data/upload` | POST | Загрузка данных (CSV/XLSX) |
        | `/api/v1/data/import/postgres` | POST | Импорт из PostgreSQL |
        | `/api/v1/data/datasets` | GET | Список датасетов |
        | `/api/v1/forecast/batch` | GET | Пакетный прогноз |
        """)

    # --- Пользователи ---
    with admin_tabs[1]:
        st.markdown("### Управление пользователями")
        st.info("Управление пользователями через CLI или REST API (требуется роль admin)")

        with st.expander("Создать пользователя (CLI)", expanded=False):
            new_email = st.text_input("Email", key="new_user_email")
            new_role = st.selectbox("Роль", ["viewer", "analyst", "admin"], key="new_user_role")
            new_pass = st.text_input("Пароль", type="password", key="new_user_pass")

            if st.button("Создать", key="btn_create_user"):
                if new_email and new_pass:
                    success, error = run_command(
                        [sys.executable, 'cli.py', 'user', 'create', new_email,
                         '-p', new_pass, '-r', new_role],
                        "Создание пользователя"
                    )
                    if success:
                        st.success(f"Пользователь {new_email} создан (роль: {new_role})")
                    else:
                        st.error(f"Ошибка: {error}")
                else:
                    st.warning("Заполните email и пароль")

        st.markdown("#### Роли системы")
        roles_data = pd.DataFrame([
            {"Роль": "admin", "Описание": "Полный доступ: пользователи, настройки, аудит",
             "Права": "read, write, admin, manage_users, view_audit"},
            {"Роль": "analyst", "Описание": "Загрузка данных, обучение моделей, экспорт",
             "Права": "read, write, train, export"},
            {"Роль": "viewer", "Описание": "Просмотр дашборда и прогнозов",
             "Права": "read"},
        ])
        st.dataframe(roles_data, use_container_width=True, hide_index=True)

    # --- Аудит ---
    with admin_tabs[2]:
        st.markdown("### Журнал аудита")
        st.info("Аудит-логи доступны через API `/api/v1/admin/audit` или CLI `forecastly audit export`")

        # Показываем логи приложения если доступны
        log_file = logs_dir / 'forecastly.log'
        if log_file.exists():
            with st.expander("Последние записи лога", expanded=True):
                log_lines_count = st.slider("Количество строк", 10, 200, 50, key="log_lines")
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    recent = lines[-log_lines_count:]
                    st.code("".join(recent), language="log")
                except Exception as e:
                    st.error(f"Ошибка чтения логов: {e}")
        else:
            st.info("Файл логов не найден")

    # --- Настройки ---
    with admin_tabs[3]:
        st.markdown("### Настройки системы")
        st.markdown("""
        Настройки задаются через переменные окружения (`.env`):
        - `SECRET_KEY` — ключ для JWT токенов
        - `USE_DATABASE` — использовать PostgreSQL (true/false)
        - `CORS_ORIGINS` — разрешённые домены для CORS
        - `RATE_LIMIT` — лимит запросов (по умолчанию 100/minute)
        - `AUTH_RATE_LIMIT` — лимит auth запросов (5/minute)
        - `LOG_LEVEL` — уровень логирования
        - `ENVIRONMENT` — окружение (development/production)
        """)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>📊 Forecastly v1.4 | Дипломный проект | 2026</p>
    <p style="font-size: 0.8rem;">Вульферт Н.Е. | Группа 122 ИСП | Новосибирский политехнический колледж</p>
</div>
""", unsafe_allow_html=True)
