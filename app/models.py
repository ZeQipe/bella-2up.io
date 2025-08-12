"""
Модели данных для работы с базой данных
Определяет структуры для хранения сообщений и состояния чатов
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

from .config import PersonaType


class MessageRole(Enum):
    """Роли сообщений в диалоге"""
    USER = "user"           # Сообщение от пользователя
    ASSISTANT = "assistant"  # Ответ бота/AI
    SYSTEM = "system"       # Системное сообщение (для промтов)


@dataclass
class Message:
    """Модель сообщения в диалоге"""
    id: Optional[int]           # ID записи в БД (None для новых)
    chat_id: int               # ID чата в Telegram
    user_id: Optional[int]     # ID пользователя в Telegram (может быть None для групп)
    message_id: Optional[int]  # ID сообщения в Telegram
    role: MessageRole          # Роль отправителя
    content: str              # Текст сообщения
    persona_type: PersonaType # Активная персона на момент сообщения
    created_at: datetime      # Время создания
    
    def to_dict(self) -> dict:
        """Преобразование в словарь для передачи в AI API"""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ChatState:
    """Состояние чата пользователя"""
    chat_id: int                    # ID чата в Telegram
    current_persona: PersonaType    # Текущая активная персона
    message_count: int             # Количество сообщений в истории
    last_activity: datetime        # Время последней активности
    
    @classmethod
    def create_default(cls, chat_id: int) -> 'ChatState':
        """Создает состояние чата по умолчанию"""
        return cls(
            chat_id=chat_id,
            current_persona=PersonaType.BUSINESS,  # По умолчанию деловой режим
            message_count=0,
            last_activity=datetime.now()
        )
