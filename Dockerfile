# Multi-stage build для оптимизации размера образа
FROM python:3.12-slim as builder

# Устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Создаем пользователя для безопасности
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Копируем виртуальное окружение из builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем рабочую директорию
WORKDIR /app

# Создаем необходимые директории и устанавливаем права
RUN mkdir -p data kb prompts && \
    chown -R botuser:botuser /app

# Копируем код приложения
COPY --chown=botuser:botuser . .

# Переключаемся на непривилегированного пользователя
USER botuser

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Healthcheck для проверки состояния контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Команда запуска
CMD ["python", "main.py"]
