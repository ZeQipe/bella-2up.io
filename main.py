"""
Основной файл запуска Telegram бота для онлайн казино
Инициализирует все компоненты и запускает бота (синхронный подход)
"""
import asyncio
import sys
import structlog

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.config import config
from app.services.vector_service import vector_service
from app.handlers.telegram_handlers import TelegramHandlers

# Настройка логирования (простой формат для отладки)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer()  # Простой вывод в консоль
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def register_handlers(app: Application) -> None:
    """Регистрирует обработчики команд и сообщений"""
    app.add_handler(CommandHandler("start", TelegramHandlers.start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, TelegramHandlers.message_handler))
    app.add_error_handler(TelegramHandlers.error_handler)
    logger.info("Handlers registered")


def main():
    """Основная функция запуска (синхронная, по порядку)"""
    logger.info("Casino Telegram Bot starting", version="1.0.0")

    try:
        # 1) Валидация конфигурации
        logger.info("Validating configuration")
        config.validate()
        logger.info("Configuration validated successfully")

        # 2) Инициализация векторной БД (синхронно)
        logger.info("Initializing knowledge base", kb_dir=config.KNOWLEDGE_BASE_DIR)
        vector_service.initialize_knowledge_base_sync(config.KNOWLEDGE_BASE_DIR)
        logger.info("Knowledge base initialized successfully")

        # 3) Запуск бота
        logger.info("Creating Telegram application")
        app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        register_handlers(app)

        logger.info("Starting bot polling")
        app.run_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]
        )

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
