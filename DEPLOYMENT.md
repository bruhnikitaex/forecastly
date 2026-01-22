# Deployment Guide для Forecastly

Руководство по развёртыванию проекта в различных окружениях.

---

## 📋 Таблица содержания

1. [Локальное развёртывание](#локальное-развёртывание)
2. [Docker развёртывание](#docker-развёртывание)
3. [Облачное развёртывание](#облачное-развёртывание)
4. [Мониторинг и логирование](#мониторинг-и-логирование)
5. [Troubleshooting](#troubleshooting)

---

## Локальное развёртывание

### Требования
- Python 3.11+
- pip / conda
- Git
- ~2 GB свободного места

### Шаг 1: Клонирование и установка
```bash
git clone https://github.com/bruhnikita/forecastly.git
cd forecastly

# Создание виртуального окружения
python -m venv venv

# Активация (выбрать в зависимости от OS)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Шаг 2: Конфигурация
```bash
# Копирование и редактирование .env
cp .env.example .env

# Отредактируй .env если нужно (опционально)
# По умолчанию используются значения из примера
```

### Шаг 3: Запуск Streamlit дашборда
```bash
streamlit run src/ui/dashboard.py
```

Откроется: **http://localhost:8501**

### Шаг 4: Запуск API (в отдельном терминале)
```bash
uvicorn src.api.app:app --reload --port 8000 --host 0.0.0.0
```

Документация: **http://localhost:8000/docs**

---

## Docker развёртывание

### Требования
- Docker 20.10+
- Docker Compose 2.0+
- ~1.5 GB свободного места для образов

### Вариант 1: Быстрый старт (рекомендуется)

```bash
# Запуск всех сервисов
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f api
docker-compose logs -f dashboard

# Остановка
docker-compose down
```

### Вариант 2: Сборка пользовательского образа

```bash
# Сборка образа API
docker build -t forecastly-api:latest .

# Сборка образа Dashboard
docker build -t forecastly-dashboard:latest -f Dockerfile.streamlit .

# Запуск API
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name forecastly-api \
  forecastly-api:latest

# Запуск Dashboard
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name forecastly-dashboard \
  forecastly-dashboard:latest
```

### Вариант 3: Production развёртывание (рекомендуется)

Создайте `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    image: forecastly-api:latest
    restart: always
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=WARNING
    volumes:
      - api_data:/app/data
      - api_logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    image: forecastly-dashboard:latest
    restart: always
    ports:
      - "8501:8501"
    depends_on:
      - api
    volumes:
      - dashboard_data:/app/data

  db:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_DB=forecastly
      - POSTGRES_USER=forecastly
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U forecastly"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  api_data:
  api_logs:
  dashboard_data:
  postgres_data:
```

Запуск:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Облачное развёртывание

### Heroku

```bash
# Логирование
heroku login

# Создание приложения
heroku create forecastly-api

# Установка buildpack для Python
heroku buildpacks:set heroku/python

# Деплой
git push heroku main

# Просмотр логов
heroku logs --tail
```

### AWS EC2

```bash
# SSH в инстанс
ssh -i key.pem ec2-user@your-instance

# Установка Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Установка Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Клонирование репозитория
git clone https://github.com/bruhnikita/forecastly.git
cd forecastly

# Запуск
docker-compose up -d
```

### Google Cloud Run (для API)

```bash
# Сборка и push в GCR
gcloud builds submit --tag gcr.io/my-project/forecastly-api

# Деплой
gcloud run deploy forecastly-api \
  --image gcr.io/my-project/forecastly-api \
  --platform managed \
  --region us-central1 \
  --port 8000

# Просмотр логов
gcloud run logs read forecastly-api --platform managed --region us-central1
```

### Azure Container Instances

```bash
# Логирование
az login

# Создание image registry
az acr create --resource-group myResourceGroup --name myRegistry --sku Basic

# Build и push
az acr build --registry myRegistry --image forecastly:latest .

# Деплой контейнера
az container create \
  --resource-group myResourceGroup \
  --name forecastly-api \
  --image myRegistry.azurecr.io/forecastly:latest \
  --cpu 2 --memory 1 \
  --ports 8000
```

---

## Мониторинг и логирование

### Логи приложения

```bash
# Docker Compose
docker-compose logs -f api          # Логи API
docker-compose logs -f dashboard    # Логи Dashboard
docker-compose logs --tail=100      # Последние 100 строк

# Локально
tail -f logs/app.log
```

### Health проверки

```bash
# API health check
curl http://localhost:8000/health

# API статус
curl http://localhost:8000/api/v1/status

# Docker Compose health
docker-compose ps  # Смотрит STATUS
```

### Мониторинг производительности

```bash
# Docker stats
docker stats

# Использование памяти
docker-compose stats
```

### Логирование в файл

```bash
# Ротация логов уже настроена в loguru
# Файлы находятся в logs/app.log
# Ротация: 1 MB, сохранение: 10 файлов
```

---

## Troubleshooting

### Проблема: Ошибка при загрузке CSV

**Симптомы:**
```
ValidationError: Отсутствуют обязательные колонки: ...
```

**Решение:**
```python
from src.etl.validate import validate_csv_file

# Проверка формата CSV
try:
    df = validate_csv_file('data/raw/sales.csv')
except Exception as e:
    print(f"Ошибка: {e}")
```

Убедись, что CSV содержит колонки: `date`, `sku_id`

### Проблема: API не отвечает

```bash
# Проверка, запущен ли контейнер
docker-compose ps

# Перезагрузка API
docker-compose restart api

# Просмотр логов ошибок
docker-compose logs api | tail -50
```

### Проблема: Streamlit не грузится

```bash
# Очистка кэша Streamlit
rm -rf ~/.streamlit/

# Перезапуск
docker-compose restart dashboard
```

### Проблема: PostgreSQL не подключается

```bash
# Проверка статуса БД
docker-compose ps db

# Проверка логов БД
docker-compose logs db

# Перезагрузка БД
docker-compose down
docker-compose up -d db
docker-compose up -d api
```

### Проблема: Много памяти используется

```bash
# Очистка неиспользуемых образов/контейнеров
docker system prune -a

# Проверка размера данных
du -sh data/

# Очистка старых логов
docker-compose exec api sh -c "rm -f logs/app.*.log"
```

### Проблема: Python версия несовместима

```bash
# Проверка версии
python --version

# Требуется Python 3.11+
# Обновление или установка новой версии требуется
```

---

## 🔒 Безопасность

### Переменные окружения

**НИКОГДА** не коммитьте `.env` файл с реальными паролями!

```bash
# В .gitignore уже есть
*.env
.env.local
```

### Для production используй:

```bash
# Secure пароль для базы данных
export DB_PASSWORD=$(openssl rand -base64 32)

# Secure API ключ (если добавишь аутентификацию)
export API_KEY=$(openssl rand -hex 32)
```

### Firewall и сетевая безопасность

```bash
# Ограничение доступа к портам (уважаемый firewall)
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8501/tcp  # Dashboard
sudo ufw allow 5432/tcp  # PostgreSQL (только для локальной сети!)
```

---

## 📊 Performance Tips

### Оптимизация Streamlit дашборда
- Используется `@st.cache_data` для загрузки данных
- Используется `@st.cache_resource` для моделей
- Дашборд работает в 10x быстрее благодаря кэшированию

### Оптимизация API
- Используется connection pooling для БД
- Кэширование ответов API (можно добавить)
- Асинхронные endpoints (можно добавить)

### Оптимизация БД
```sql
-- Индексы для ускорения запросов
CREATE INDEX idx_predictions_sku ON predictions(sku_id);
CREATE INDEX idx_predictions_date ON predictions(date);
```

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверь [Troubleshooting](#troubleshooting) раздел
2. Посмотри логи: `docker-compose logs -f`
3. Открой Issue на GitHub
4. Напиши в контакты

---

**Версия**: 1.0  
**Последнее обновление**: 2025-11-11
