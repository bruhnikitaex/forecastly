# Forecastly API - Usage Examples

This directory contains examples of how to integrate with the Forecastly API using different programming languages and tools.

## Available Examples

### 1. Python Example (`python_example.py`)

Full-featured Python client demonstrating all API capabilities.

**Requirements:**
```bash
pip install requests pandas
```

**Usage:**
```bash
python examples/python_example.py
```

**Features:**
- ✅ Object-oriented client class
- ✅ Type hints
- ✅ Error handling
- ✅ Batch processing
- ✅ Data export to CSV
- ✅ Comprehensive examples

---

### 2. JavaScript Example (`javascript_example.js`)

Node.js example using axios for API requests.

**Requirements:**
```bash
npm install axios
```

**Usage:**
```bash
node examples/javascript_example.js
```

**Features:**
- ✅ Async/await syntax
- ✅ Promise-based
- ✅ Error handling
- ✅ Batch processing
- ✅ Clean class structure

---

### 3. cURL Examples (`curl_examples.sh`)

Shell script with curl commands for all API endpoints.

**Requirements:**
- `curl`
- `jq` (for JSON parsing)

**Usage:**
```bash
chmod +x examples/curl_examples.sh
./examples/curl_examples.sh
```

**Features:**
- ✅ All API endpoints covered
- ✅ CSV export examples
- ✅ Error handling demonstrations
- ✅ Batch processing
- ✅ Formatted output with jq

---

## Quick Start

1. **Start the Forecastly API:**
   ```bash
   # Local
   uvicorn src.api.app:app --reload --port 8000

   # Or Docker
   docker-compose up -d
   ```

2. **Run an example:**
   ```bash
   # Python
   python examples/python_example.py

   # JavaScript
   node examples/javascript_example.js

   # cURL
   ./examples/curl_examples.sh
   ```

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/skus` | GET | Get available SKUs |
| `/api/v1/predict` | GET | Get forecast for SKU |
| `/api/v1/predict/rebuild` | POST | Rebuild forecasts |
| `/api/v1/metrics` | GET | Get model metrics |
| `/api/v1/status` | GET | Get system status |

Full API documentation: [docs/api.md](../docs/api.md)

---

## Integration Patterns

### Pattern 1: Simple Forecast Retrieval

**Python:**
```python
import requests

response = requests.get('http://localhost:8000/api/v1/predict', params={
    'sku_id': 'SKU001',
    'horizon': 14
})
forecast = response.json()
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14');
const forecast = await response.json();
```

**cURL:**
```bash
curl "http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14"
```

---

### Pattern 2: Batch Processing

**Python:**
```python
client = ForecastlyClient()
skus = client.get_skus()

for sku in skus:
    forecast = client.get_forecast(sku, horizon=7)
    # Process forecast...
```

**JavaScript:**
```javascript
const skus = await client.getSKUs();

for (const sku of skus) {
    const forecast = await client.getForecast(sku, 7);
    // Process forecast...
}
```

---

### Pattern 3: Export to CSV

**Python:**
```python
import pandas as pd

forecast = client.get_forecast('SKU001', 14)
df = pd.DataFrame(forecast['predictions'])
df.to_csv('forecast.csv', index=False)
```

**cURL + jq:**
```bash
curl -s "http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14" | \
    jq -r '.predictions[] | [.date, .ensemble] | @csv' > forecast.csv
```

---

## Integration with BI Tools

### Power BI

Use the Python example as a base:

1. Open Power BI Desktop
2. Get Data → Python script
3. Paste the following:

```python
import requests
import pandas as pd

response = requests.get('http://localhost:8000/api/v1/predict', params={
    'sku_id': 'SKU001',
    'horizon': 30
})
forecast = pd.DataFrame(response.json()['predictions'])
```

### Excel (Power Query)

1. Data → From Web
2. Enter URL: `http://localhost:8000/api/v1/predict?sku_id=SKU001&horizon=14`
3. Parse JSON response

### Tableau

1. Connect to Data → Web Data Connector
2. Use the API URL
3. Configure refresh schedule

---

## Error Handling

All examples include proper error handling. The API returns structured error responses:

```json
{
  "error": {
    "code": "DATA_NOT_FOUND",
    "message": "SKU 'INVALID' not found",
    "details": {
      "sku_id": "INVALID",
      "available_skus": ["SKU001", "SKU002"]
    }
  },
  "path": "/api/v1/predict",
  "timestamp": "2025-01-25T10:30:00.123456"
}
```

**Handle errors appropriately:**

```python
try:
    forecast = client.get_forecast('INVALID')
except requests.HTTPError as e:
    if e.response.status_code == 404:
        print("SKU not found")
    else:
        print(f"API error: {e}")
```

---

## Performance Tips

1. **Use batch processing** for multiple SKUs instead of individual requests
2. **Cache results** when appropriate (forecasts don't change frequently)
3. **Use async/await** in JavaScript for parallel requests
4. **Set timeouts** to avoid hanging on slow requests
5. **Implement retry logic** for transient errors

---

## Authentication

If authentication is enabled (`USE_DATABASE=true`):

```python
# Login
response = requests.post('http://localhost:8000/api/v1/auth/login', data={
    'username': 'user@example.com',
    'password': 'password'
})
token = response.json()['access_token']

# Use token
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/v1/predict',
                        headers=headers,
                        params={'sku_id': 'SKU001'})
```

---

## Support

For questions or issues:
- Documentation: [README.md](../README.md)
- API Docs: [docs/api.md](../docs/api.md)
- Issues: [GitHub Issues](https://github.com/bruhnikita/forecastly/issues)

---

**Happy integrating!** 🚀
