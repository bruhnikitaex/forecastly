FROM python:3.11-slim

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание директорий для данных и логов
RUN mkdir -p data/raw data/processed data/models logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py || exit 1

# Exposing API порт
EXPOSE 8000

# Запуск API сервера по умолчанию
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
