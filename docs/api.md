# Forecastly API Documentation

**Version:** 1.1.0
**Base URL:** `http://localhost:8000`

API предоставляет программный доступ к системе анализа и прогнозирования продаж.
Все endpoints версионированы с префиксом `/api/v1/`.

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Root](#root)
  - [System Status](#system-status)
  - [SKU Management](#sku-management)
  - [Predictions](#predictions)
  - [Metrics](#metrics)
  - [Database](#database)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Authentication

> **Note:** Authentication endpoints доступны только при включенной базе данных (`USE_DATABASE=true`).

### Register User

**POST** `/api/v1/auth/register`

Регистрация нового пользователя.

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "SecurePassword123",
  "full_name": "Ivan Ivanov"
}
```

**Response:**
```json
{
  "id": 1,
  "username": "user@example.com",
  "full_name": "Ivan Ivanov",
  "is_active": true,
  "is_admin": false,
  "created_at": "2025-01-25T10:30:00"
}
```

### Login

**POST** `/api/v1/auth/login`

Вход в систему и получение access token.

**Request Body (form-data):**
```
username=user@example.com
password=SecurePassword123
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePassword123"
```

---

## Rate Limiting

API защищен rate limiting для предотвращения злоупотреблений:

- **General endpoints:** 100 запросов/минута (настраивается через `RATE_LIMIT`)
- **Auth endpoints:** 5 запросов/минута (настраивается через `AUTH_RATE_LIMIT`)

При превышении лимита API вернет `429 Too Many Requests`.

---

## Endpoints

### Health Check

**GET** `/health`

Проверка работоспособности API.

**Response:**
```json
{
  "status": "ok",
  "service": "forecastly-api",
  "version": "1.1.0",
  "timestamp": "2025-01-25T10:30:00.123456",
  "database_mode": true,
  "database_connected": true
}
```

**cURL Example:**
```bash
curl http://localhost:8000/health
```

---

### Root

**GET** `/`

Получение информации об API и доступных endpoints.

**Response:**
```json
{
  "service": "forecastly-api",
  "version": "1.1.0",
  "database_mode": true,
  "docs": "/docs",
  "redoc": "/redoc",
  "endpoints": {
    "health": "/health",
    "skus": "/api/v1/skus",
    "predict": "/api/v1/predict?sku_id=SKU001&horizon=14",
    "rebuild": "/api/v1/predict/rebuild?horizon=14",
    "metrics": "/api/v1/metrics",
    "status": "/api/v1/status",
    "db_stats": "/api/v1/db/stats",
    "forecast_runs": "/api/v1/forecast-runs"
  },
  "timestamp": "2025-01-25T10:30:00.123456"
}
```

---

### System Status

**GET** `/api/v1/status`

Получение статуса системы и доступности данных.

**Response:**
```json
{
  "system": "ready",
  "timestamp": "2025-01-25T10:30:00.123456",
  "database_mode": true,
  "data_available": {
    "raw": true,
    "processed": true,
    "predictions": true,
    "metrics": true,
    "models": {
      "prophet": true,
      "xgboost": true
    }
  },
  "database": {
    "total_skus": 50,
    "total_predictions": 700,
    "total_forecast_runs": 5,
    "total_users": 3
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/status
```

---

### SKU Management

#### Get All SKUs

**GET** `/api/v1/skus`

Получение списка всех доступных SKU.

**Response:**
```json
{
  "skus": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
  "count": 5
}
```

**Python Example:**
```python
import requests

response = requests.get('http://localhost:8000/api/v1/skus')
data = response.json()
print(f"Available SKUs: {data['skus']}")
print(f"Total count: {data['count']}")
```

---

### Predictions

#### Get Forecast

**GET** `/api/v1/predict`

Получение прогноза для конкретного SKU.

**Query Parameters:**
- `sku_id` (required): SKU товара (например, `SKU001`)
- `horizon` (optional): Горизонт прогноза в днях (1-120, default: 14)

**Response:**
```json
{
  "sku_id": "SKU001",
  "horizon": 14,
  "count": 14,
  "source": "database",
  "predictions": [
    {
      "date": "2025-01-26",
      "prophet": 22.3,
      "xgb": 20.8,
      "ensemble": 21.6,
      "p_low": 18.5,
      "p_high": 26.1
    },
    {
      "date": "2025-01-27",
      "prophet": 23.1,
      "xgb": 22.0,
      "ensemble": 22.6,
      "p_low": 19.2,
      "p_high": 27.0
    }
  ]
}
```

**Error Responses:**
- `400 Bad Request`: Отсутствует обязательный параметр `sku_id`
- `404 Not Found`: Прогноз для указанного SKU не найден
- `422 Unprocessable Entity`: Неверное значение параметра `horizon`

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14"
```

**Python Example:**
```python
import requests

params = {
    'sku_id': 'SKU001',
    'horizon': 14
}
response = requests.get('http://localhost:8000/api/v1/predict', params=params)
forecast = response.json()

for prediction in forecast['predictions']:
    print(f"{prediction['date']}: {prediction['ensemble']:.1f} units")
```

#### Rebuild Forecasts

**POST** `/api/v1/predict/rebuild`

Пересчет прогнозов для всех SKU.

**Query Parameters:**
- `horizon` (optional): Горизонт прогноза (1-120, default: 14)
- `save_to_db` (optional): Сохранить результаты в БД (default: false)

**Response:**
```json
{
  "status": "ok",
  "message": "Прогнозы пересчитаны успешно",
  "horizon": 14,
  "timestamp": "2025-01-25T10:35:00.123456",
  "saved_to_db": true,
  "records_saved": 700,
  "run_id": "20250125103500"
}
```

**Error Responses:**
- `500 Internal Server Error`: Ошибка при выполнении прогнозирования
- `504 Gateway Timeout`: Превышено время ожидания (>5 минут)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/rebuild?horizon=14&save_to_db=true"
```

---

### Metrics

**GET** `/api/v1/metrics`

Получение метрик качества прогнозирования.

**Response:**
```json
{
  "count": 5,
  "source": "database",
  "metrics": [
    {
      "sku_id": "SKU001",
      "mape_prophet": 5.2,
      "mape_xgboost": 4.8,
      "mape_naive": 8.3,
      "mape_ens": 4.5,
      "best_model": "ens"
    },
    {
      "sku_id": "SKU002",
      "mape_prophet": 6.1,
      "mape_xgboost": 5.5,
      "mape_naive": 9.0,
      "mape_ens": 5.0,
      "best_model": "ens"
    }
  ]
}
```

**Python Example:**
```python
import requests
import pandas as pd

response = requests.get('http://localhost:8000/api/v1/metrics')
data = response.json()

# Преобразование в DataFrame
df = pd.DataFrame(data['metrics'])
print(df[['sku_id', 'mape_ens', 'best_model']])

# Средняя точность Ensemble
avg_mape = df['mape_ens'].mean()
print(f"Average Ensemble MAPE: {avg_mape:.1f}%")
```

---

### Database

> **Note:** Доступно только при `USE_DATABASE=true`

#### Database Statistics

**GET** `/api/v1/db/stats`

Получение статистики базы данных.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-25T10:30:00.123456",
  "stats": {
    "total_skus": 50,
    "total_predictions": 700,
    "total_forecast_runs": 5,
    "total_users": 3
  }
}
```

#### Forecast Runs

**GET** `/api/v1/forecast-runs`

Получение истории запусков прогнозирования.

**Query Parameters:**
- `skip` (optional): Пропустить первые N записей (default: 0)
- `limit` (optional): Максимальное количество записей (1-100, default: 20)

**Response:**
```json
{
  "count": 3,
  "runs": [
    {
      "run_id": "20250125103500",
      "horizon": 14,
      "model_type": "ensemble",
      "status": "completed",
      "started_at": "2025-01-25T10:35:00",
      "completed_at": "2025-01-25T10:37:30",
      "records_count": 700,
      "error_message": null
    }
  ]
}
```

#### Sync CSV to Database

**POST** `/api/v1/db/sync`

Синхронизация данных из CSV файлов в базу данных.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-25T10:40:00.123456",
  "synced": {
    "predictions": 700,
    "metrics": 50
  }
}
```

---

## Error Handling

API использует стандартные HTTP коды состояния:

| Code | Description |
|------|-------------|
| 200 | OK - Запрос выполнен успешно |
| 400 | Bad Request - Неверные параметры запроса |
| 401 | Unauthorized - Требуется аутентификация |
| 403 | Forbidden - Доступ запрещен |
| 404 | Not Found - Ресурс не найден |
| 422 | Unprocessable Entity - Ошибка валидации |
| 429 | Too Many Requests - Превышен rate limit |
| 500 | Internal Server Error - Внутренняя ошибка сервера |
| 504 | Gateway Timeout - Превышено время ожидания |

**Error Response Format:**
```json
{
  "detail": "Описание ошибки на русском языке"
}
```

---

## Examples

### Integration with Power BI

```python
import pandas as pd
import requests

# Получить метрики
response = requests.get('http://localhost:8000/api/v1/metrics')
df_metrics = pd.DataFrame(response.json()['metrics'])

# Получить прогноз для всех SKU
skus_response = requests.get('http://localhost:8000/api/v1/skus')
skus = skus_response.json()['skus']

forecasts = []
for sku in skus:
    response = requests.get(f'http://localhost:8000/api/v1/predict?sku_id={sku}&horizon=30')
    if response.status_code == 200:
        data = response.json()
        for pred in data['predictions']:
            forecasts.append({
                'sku_id': sku,
                'date': pred['date'],
                'forecast': pred['ensemble']
            })

df_forecasts = pd.DataFrame(forecasts)
```

### Integration with Excel (VBA)

```vba
Sub GetForecast()
    Dim http As Object
    Dim url As String
    Dim response As String

    Set http = CreateObject("MSXML2.XMLHTTP")
    url = "http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14"

    http.Open "GET", url, False
    http.send

    response = http.responseText
    ' Парсинг JSON и вставка в Excel
End Sub
```

### Integration with 1C

```bsl
Функция ПолучитьПрогноз(КодТовара, ГоризонтДней)
    HTTPСоединение = Новый HTTPСоединение("localhost", 8000);
    HTTPЗапрос = Новый HTTPЗапрос("/api/v1/predict?sku_id=" + КодТовара + "&horizon=" + ГоризонтДней);

    HTTPОтвет = HTTPСоединение.Получить(HTTPЗапрос);
    Если HTTPОтвет.КодСостояния = 200 Тогда
        ЧтениеJSON = Новый ЧтениеJSON;
        ЧтениеJSON.УстановитьСтроку(HTTPОтвет.ПолучитьТелоКакСтроку());
        Результат = ПрочитатьJSON(ЧтениеJSON);
        Возврат Результат;
    КонецЕсли;
КонецФункции
```

---

## Interactive Documentation

FastAPI предоставляет автоматически сгенерированную интерактивную документацию:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

---

## Support

Для вопросов и issue:
- GitHub: https://github.com/bruhnikita/forecastly/issues
- Email: support@forecastly.local

**Автор:** Вульферт Никита Евгеньевич
**Группа:** 122 ИСП
**Год:** 2025
