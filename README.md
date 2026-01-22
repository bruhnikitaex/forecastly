# Forecastly — система анализа и прогнозирования продаж

> Дипломный проект студента **Вульферта Никиты Евгеньевича**, группа **122 ИСП**
> Новосибирский политехнический колледж, 2026 год

---

## О проекте

**Forecastly** — это система, предназначенная для анализа и прогнозирования продаж компании.
Решение объединяет в себе **ETL-обработку данных**, **машинное обучение (Prophet и XGBoost)**,
**интерактивный дашборд на Streamlit** и **REST API на FastAPI**.

### Основные возможности:
- Загрузка, очистка и валидация данных о продажах
- Анализ динамики спроса и ключевых показателей
- Обучение и сравнение моделей прогнозирования (Prophet, XGBoost, Ensemble)
- Визуализация прогнозов с интерактивными графиками
- Экспорт метрик и истории прогнозов
- REST API для интеграции с внешними системами (Power BI, Excel, 1C)
- Docker контейнеризация для простого развёртывания
- Оптимизированное кэширование данных в Streamlit

---

## Архитектура проекта

```
forecastly/
├── src/
│   ├── api/                    # FastAPI REST-сервис
│   │   ├── app.py             # Основной API с versioning (/api/v1/...)
│   │   └── schemas.py         # Pydantic схемы
│   ├── etl/                    # ETL-конвейер
│   │   ├── load_data.py       # Загрузка CSV
│   │   ├── clean_data.py      # Очистка и нормализация
│   │   ├── validate.py        # Валидация входных данных
│   │   ├── feature_builder.py # Построение признаков
│   │   └── ...
│   ├── models/                 # ML модели
│   │   ├── train_xgboost.py   # Обучение XGBoost
│   │   ├── train_prophet.py   # Обучение Prophet
│   │   ├── predict.py         # Прогнозирование
│   │   └── evaluate.py        # Оценка качества
│   ├── ui/                     # Streamlit дашборд
│   │   └── dashboard.py       # Интерактивный интерфейс
│   └── utils/                  # Утилиты
│       ├── config.py          # Конфиг с поддержкой .env
│       ├── logger.py          # Логирование
│       └── helpers.py         # Вспомогательные функции
├── data/
│   ├── raw/                    # Сырые данные
│   ├── processed/              # Обработанные данные
│   └── models/                 # Сохранённые модели
├── configs/
│   ├── paths.yaml             # Пути к файлам
│   └── model.yaml             # Параметры моделей
├── tests/                      # Тесты
│   ├── test_api.py            # Тесты API
│   ├── test_etl.py            # Тесты ETL
│   └── test_model.py          # Тесты моделей
├── docs/
│   └── api.md                 # Документация API
├── Dockerfile                 # Docker образ API
├── Dockerfile.streamlit       # Docker образ дашборда
├── docker-compose.yml         # Docker Compose (API + Dashboard + DB)
├── requirements.txt           # Python зависимости
└── README.md
```

---

## Запуск проекта

### Вариант 1: Локальный запуск (без Docker)

#### 1. Установка зависимостей
```bash
# Клонирование
git clone https://github.com/bruhnikita/forecastly.git
cd forecastly

# Создание виртуального окружения
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt
```

#### 2. Запуск Streamlit дашборда
```bash
streamlit run src/ui/dashboard.py
```
Откроется: **http://localhost:8501**

#### 3. Запуск API сервера (в отдельном терминале)
```bash
uvicorn src.api.app:app --reload --port 8000
```
Документация: **http://localhost:8000/docs**

---

### Вариант 2: Запуск через Docker Compose (рекомендуется для production)

```bash
# Запуск всех сервисов (API + Dashboard + PostgreSQL)
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f api
docker-compose logs -f dashboard

# Остановка
docker-compose down
```

**Доступные сервисы:**
- **API**: http://localhost:8000 (документация: /docs)
- **Dashboard**: http://localhost:8501
- **PostgreSQL**: localhost:5432 (для будущей интеграции)

---

## REST API

Все endpoints версионированы с префиксом `/api/v1/`.

### Основные endpoints:

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Health check |
| `/api/v1/skus` | GET | Список доступных SKU |
| `/api/v1/predict` | GET | Получить прогноз по SKU |
| `/api/v1/predict/rebuild` | POST | Пересчитать прогнозы |
| `/api/v1/metrics` | GET | Получить метрики качества |
| `/api/v1/status` | GET | Статус системы |

### Пример использования:
```python
import requests

# Получить список SKU
response = requests.get('http://localhost:8000/api/v1/skus')
print(response.json())

# Получить прогноз
forecast = requests.get('http://localhost:8000/api/v1/predict',
                        params={'sku_id': 'SKU001', 'horizon': 14})
print(forecast.json())
```

Полная документация: [docs/api.md](docs/api.md)

---

## Тестирование

```bash
# Запуск всех тестов
pytest tests/ -v

# Запуск с покрытием
pytest tests/ -v --cov=src --cov-report=html
```

---

## Используемые технологии

| Область | Инструменты |
|---------|-------------|
| Язык программирования | Python 3.11 |
| Веб-интерфейс | Streamlit |
| API | FastAPI + Uvicorn |
| Машинное обучение | Prophet, XGBoost, scikit-learn |
| Аналитика данных | Pandas, NumPy, Matplotlib, Plotly |
| Логирование | Loguru |
| Контейнеризация | Docker, Docker Compose |
| Тестирование | pytest |

---

## Пример работы

Пример прогноза в дашборде:

| Дата | Факт | Prophet | XGBoost | Ensemble |
|------|------|---------|---------|----------|
| 2025-10-30 | 21 | 22.3 | 20.8 | 21.6 |
| 2025-10-31 | 24 | 23.1 | 22.0 | 22.6 |
| 2025-11-01 | 23 | 24.4 | 22.8 | 23.6 |

---

## Автор

**Вульферт Никита Евгеньевич**
Группа 122 ИСП, Новосибирский политехнический колледж
Руководитель проекта: Гритчин И.В.

---

## Лицензия

Проект выполнен в образовательных целях.
Все материалы можно использовать для учебных и исследовательских проектов.
