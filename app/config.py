"""
Конфигурация приложения
Централизованное управление всеми настройками через переменные окружения
"""
import os
from enum import Enum
from typing import List
from dotenv import load_dotenv
import os as _os

# Загружаем переменные окружения из .env файла если есть
load_dotenv()


class PersonaType(Enum):
    """Типы персон бота для разных стилей общения"""
    BUSINESS = "business"  # Деловой режим - строгий техподдержка
    BELLA = "bella"        # Женская персона с легким флиртом
    BEN = "ben"           # Мужская персона с легким флиртом


class Config:
    """Основная конфигурация приложения"""
    
    # Telegram Bot API
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # База данных SQLite
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/bot.db")
    
    # Redis для кеширования контекста
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "3600"))  # 1 час
    
    # Лимиты истории сообщений
    MESSAGE_HISTORY_LIMIT: int = int(os.getenv("MESSAGE_HISTORY_LIMIT", "20"))
    
    # DeepSeek API (будет добавлено позже)
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # OpenAI API для эмбеддингов (будет добавлено позже)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Пути к данным
    PROMPTS_DIR: str = os.getenv("PROMPTS_DIR", "prompts")
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "kb")
    PROMOTIONS_FILE: str = os.getenv("PROMOTIONS_FILE", os.path.join(PROMPTS_DIR, "promotions.txt"))
    
    # Векторная база данных ChromaDB (будет добавлено позже)
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "data/chroma")
    
    # Пути к файлам аудита/логов (отдельно для DeepSeek и векторного поиска)
    # Для обратной совместимости используется AUDIT_LOG_PATH, если указана
    AUDIT_DEEPSEEK_LOG_PATH: str = os.getenv(
        "AUDIT_DEEPSEEK_LOG_PATH",
        os.getenv("AUDIT_LOG_PATH", _os.path.join("logs", "deepseek_audit.log"))
    )
    AUDIT_VECTOR_LOG_PATH: str = os.getenv(
        "AUDIT_VECTOR_LOG_PATH",
        os.getenv("AUDIT_LOG_PATH", _os.path.join("logs", "vector_search.log"))
    )
    
    # Сообщение при ошибках API
    ERROR_MESSAGE: str = "Оператор не в сети, просим набраться терпения"
    
    # Настройки векторного поиска
    VECTOR_SEARCH_LIMIT: int = int(os.getenv("VECTOR_SEARCH_LIMIT", "8"))
    VECTOR_SIMILARITY_THRESHOLD: float = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))
    VECTOR_REBUILD_ON_CHANGES: bool = os.getenv("VECTOR_REBUILD_ON_CHANGES", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """Валидация обязательных настроек"""
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    @classmethod
    def get_persona_buttons(cls) -> List[str]:
        """Возвращает список кнопок для выбора персоны (EN)"""
        return ["Business mode", "Bella", "Ben"]
    
    @classmethod
    def get_persona_type(cls, button_text: str) -> PersonaType:
        """Преобразует текст кнопки в тип персоны"""
        mapping = {
            "Business mode": PersonaType.BUSINESS,
            "Bella": PersonaType.BELLA,
            "Ben": PersonaType.BEN,
        }
        return mapping.get(button_text, PersonaType.BUSINESS)


# Глобальный экземпляр конфигурации
config = Config()
