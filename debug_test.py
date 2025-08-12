#!/usr/bin/env python3
"""
Отладочный тест для поиска места зависания
"""
import asyncio
import sys
import structlog
from app.config import config

# Настройка логирования
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_step_by_step():
    """Пошаговый тест каждого компонента"""
    
    print("=== ОТЛАДОЧНЫЙ ТЕСТ ===")
    
    # Шаг 1: Конфигурация
    print("1. Проверяем конфигурацию...")
    try:
        logger.info("Testing config validation")
        config.validate()
        print("✅ Конфигурация OK")
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return
    
    # Шаг 2: Импорт векторного сервиса
    print("2. Импортируем векторный сервис...")
    try:
        from app.services.vector_service import vector_service
        print("✅ Импорт векторного сервиса OK")
    except Exception as e:
        print(f"❌ Ошибка импорта векторного сервиса: {e}")
        return
    
    # Шаг 3: Инициализация ChromaDB (без векторов)
    print("3. Инициализируем ChromaDB...")
    try:
        vector_service._initialize_client()
        print("✅ ChromaDB инициализация OK")
    except Exception as e:
        print(f"❌ Ошибка ChromaDB: {e}")
        return
    
    # Шаг 4: Проверка OpenAI
    print("4. Проверяем OpenAI API...")
    try:
        from app.services.embedding_service import embedding_service
        result = await embedding_service.health_check()
        if result:
            print("✅ OpenAI API OK")
        else:
            print("❌ OpenAI API недоступен")
            return
    except Exception as e:
        print(f"❌ Ошибка OpenAI API: {e}")
        return
    
    # Шаг 5: Создание одного эмбеддинга
    print("5. Создаём тестовый эмбеддинг...")
    try:
        embedding = await embedding_service.create_embedding("Test text")
        print(f"✅ Тестовый эмбеддинг создан, размерность: {len(embedding)}")
    except Exception as e:
        print(f"❌ Ошибка создания эмбеддинга: {e}")
        return
    
    print("🎉 Все базовые компоненты работают!")
    print("Проблема скорее всего в массовой обработке файлов")

if __name__ == "__main__":
    try:
        asyncio.run(test_step_by_step())
    except KeyboardInterrupt:
        print("\n⏹️  Тест прерван пользователем")
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        sys.exit(1)
