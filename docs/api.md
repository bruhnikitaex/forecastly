# API документация (Forecastly)

Этот API предоставляет программный доступ к системе анализа и прогнозирования продаж.
API реализован на **FastAPI** и возвращает данные в формате **JSON**.

Базовый URL (локально):  
`http://127.0.0.1:8000/`

---

## 1. Проверка доступности сервиса

**GET** `/health`

**Описание:** возвращает статус работы сервиса.

**Пример запроса:**

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
Пример ответа:

json
Копировать код
{
  "status": "ok",
  "service": "forecastly-api"
}
2. Получить список доступных SKU
GET /skus

Описание: возвращает список товаров (SKU), для которых есть данные/прогноз.

Пример запроса:

http
Копировать код
GET /skus HTTP/1.1
Host: 127.0.0.1:8000
Пример ответа:

json
Копировать код
{
  "skus": ["SKU001", "SKU002", "SKU003", "SKU004"]
}
Пояснения:

список берётся из актуального датасета (data/raw/...) или из прогноза (data/processed/predictions.csv);

формат SKU нормализован в виде SKUxxx.

3. Получить прогноз по SKU
GET /predict

Параметры (query):

sku_id — обязательный, идентификатор товара, например: SKU001

horizon — необязательный, горизонт прогноза в днях, по умолчанию 14

Пример запроса:

http
Копировать код
GET /predict?sku_id=SKU001&horizon=14
Host: 127.0.0.1:8000
Пример ответа:

json
Копировать код
{
  "sku_id": "SKU001",
  "horizon": 14,
  "predictions": [
    {
      "date": "2025-10-31",
      "prophet": 22.4,
      "lgbm": 21.0,
      "ensemble": 21.7
    },
    {
      "date": "2025-11-01",
      "prophet": 23.1,
      "lgbm": 20.6,
      "ensemble": 21.9
    }
  ]
}
Что важно:

если по этому SKU нет строк в data/processed/predictions.csv, API должно вернуть 404 с понятным сообщением

если в модели появились NaN/Infinity, их нужно чистить (в текущем проекте мы заменяем их на null / убираем строку)

Возможные ответы:

200 OK — прогноз найден

400 Bad Request — не передан sku_id

404 Not Found — по такому sku_id прогноз отсутствует

500 Internal Server Error — ошибка при чтении моделей / файла

4. Пересчитать прогноз
(опциональная ручка, можно оставить как «будет реализовано»)

POST /predict/rebuild

Описание: запускает пересчёт прогноза на сервере и перезаписывает файл data/processed/predictions.csv.

Пример запроса:

http
Копировать код
POST /predict/rebuild HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "horizon": 14
}
Пример ответа:

json
Копировать код
{
  "status": "ok",
  "message": "predictions recalculated",
  "horizon": 14
}
5. Получить метрики моделей
GET /metrics

Описание: возвращает рассчитанные метрики прогноза по каждому SKU на основе файла data/processed/metrics.csv.

Пример запроса:

http
Копировать код
GET /metrics HTTP/1.1
Host: 127.0.0.1:8000
Пример ответа:

json
Копировать код
{
  "metrics": [
    {
      "sku_id": "SKU001",
      "mape_prophet": 8.4,
      "mape_lgbm": 6.9,
      "mape_naive": 15.2,
      "mape_ens": 6.5,
      "best_model": "ens"
    },
    {
      "sku_id": "SKU002",
      "mape_prophet": 12.1,
      "mape_lgbm": 9.3,
      "mape_naive": 18.0,
      "mape_ens": 9.0,
      "best_model": "lgbm"
    }
  ]
}
Пояснения:

если файла data/processed/metrics.csv нет — вернуть 404;

эту ручку можно дергать из внешнего BI (Power BI, Excel PowerQuery).

6. Структура проекта (для API)
text
Копировать код
forecastly/
├── src/
│   ├── api/
│   │   └── app.py        ← тут FastAPI
│   ├── models/           ← обучение и прогноз
│   ├── etl/              ← загрузка/очистка/фичи
│   └── ui/               ← Streamlit
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── docs/
    └── api.md            ← этот файл
7. Требования
Python 3.11+

FastAPI

Uvicorn

Запуск:

bash
Копировать код
uvicorn src.api.app:app --reload --port 8000
После запуска интерактивная документация FastAPI будет доступна по адресу:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc