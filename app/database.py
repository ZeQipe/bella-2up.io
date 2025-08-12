"""
Слой работы с базой данных SQLite
Реализует операции с сообщениями и состояниями чатов
"""
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Generator
import structlog

from .config import config, PersonaType
from .models import Message, MessageRole, ChatState

logger = structlog.get_logger()


class DatabaseManager:
    """Менеджер базы данных для работы с SQLite"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self) -> None:
        """Создает директорию для базы данных если не существует"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info("Created database directory", path=db_dir)
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Контекстный менеджер для работы с соединением БД"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Доступ к колонкам по именам
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Инициализация схемы базы данных"""
        with self.get_connection() as conn:
            # Включаем WAL режим для лучшей производительности
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Таблица для хранения сообщений
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER,
                    message_id INTEGER,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    persona_type TEXT NOT NULL CHECK(persona_type IN ('business', 'bella', 'ben')),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица для хранения состояний чатов
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_states (
                    chat_id INTEGER PRIMARY KEY,
                    current_persona TEXT NOT NULL CHECK(current_persona IN ('business', 'bella', 'ben')),
                    message_count INTEGER NOT NULL DEFAULT 0,
                    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Индексы для оптимизации запросов
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)")
            
            logger.info("Database initialized", db_path=self.db_path)
    
    def save_message(self, message: Message) -> int:
        """Сохраняет сообщение в базу данных"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO messages (chat_id, user_id, message_id, role, content, persona_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message.chat_id,
                message.user_id,
                message.message_id,
                message.role.value,
                message.content,
                message.persona_type.value,
                message.created_at
            ))
            
            message_id = cursor.lastrowid
            
            # Обновляем счетчик сообщений в состоянии чата
            self._update_message_count(conn, message.chat_id)
            
            # Очищаем старые сообщения если превышен лимит
            self._cleanup_old_messages(conn, message.chat_id)
            
            logger.debug("Message saved", 
                        message_id=message_id, 
                        chat_id=message.chat_id, 
                        role=message.role.value)
            
            return message_id
    
    def get_chat_history(self, chat_id: int, limit: int = None, hours_back: int = None) -> List[Message]:
        """
        Получает историю сообщений чата с фильтрацией по времени
        
        Args:
            chat_id: ID чата
            limit: Максимальное количество сообщений
            hours_back: Количество часов назад для фильтрации (None = без фильтрации по времени)
        """
        limit = limit or config.MESSAGE_HISTORY_LIMIT
        
        with self.get_connection() as conn:
            # Базовый запрос
            query = """
                SELECT id, chat_id, user_id, message_id, role, content, persona_type, created_at
                FROM messages
                WHERE chat_id = ?
            """
            params = [chat_id]
            
            # Добавляем фильтрацию по времени если указано
            if hours_back is not None:
                # Вычисляем timestamp для фильтрации
                cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
                query += " AND created_at >= datetime(?, 'unixepoch')"
                params.append(cutoff_time)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            messages = []
            for row in cursor.fetchall():
                message = Message(
                    id=row['id'],
                    chat_id=row['chat_id'],
                    user_id=row['user_id'],
                    message_id=row['message_id'],
                    role=MessageRole(row['role']),
                    content=row['content'],
                    persona_type=PersonaType(row['persona_type']),
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                messages.append(message)
            
            # Возвращаем в хронологическом порядке
            messages.reverse()
            
            logger.debug("Retrieved chat history", 
                        chat_id=chat_id, 
                        message_count=len(messages),
                        hours_back=hours_back)
            
            return messages
    
    def clear_chat_history(self, chat_id: int) -> int:
        """Очищает историю сообщений чата"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            deleted_count = cursor.rowcount
            
            # Сбрасываем счетчик сообщений
            conn.execute("""
                UPDATE chat_states 
                SET message_count = 0, last_activity = CURRENT_TIMESTAMP
                WHERE chat_id = ?
            """, (chat_id,))
            
            logger.info("Chat history cleared", 
                       chat_id=chat_id, 
                       deleted_messages=deleted_count)
            
            return deleted_count
    
    def get_chat_state(self, chat_id: int) -> ChatState:
        """Получает состояние чата"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT chat_id, current_persona, message_count, last_activity
                FROM chat_states
                WHERE chat_id = ?
            """, (chat_id,))
            
            row = cursor.fetchone()
            
            if row:
                return ChatState(
                    chat_id=row['chat_id'],
                    current_persona=PersonaType(row['current_persona']),
                    message_count=row['message_count'],
                    last_activity=datetime.fromisoformat(row['last_activity'])
                )
            else:
                # Создаем новое состояние чата
                state = ChatState.create_default(chat_id)
                self.save_chat_state(state)
                return state
    
    def save_chat_state(self, state: ChatState) -> None:
        """Сохраняет состояние чата"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chat_states 
                (chat_id, current_persona, message_count, last_activity)
                VALUES (?, ?, ?, ?)
            """, (
                state.chat_id,
                state.current_persona.value,
                state.message_count,
                state.last_activity
            ))
            
            logger.debug("Chat state saved", 
                        chat_id=state.chat_id, 
                        persona=state.current_persona.value)
    
    def update_chat_persona(self, chat_id: int, persona: PersonaType) -> None:
        """Обновляет персону чата"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chat_states 
                (chat_id, current_persona, message_count, last_activity)
                VALUES (
                    ?, 
                    ?, 
                    COALESCE((SELECT message_count FROM chat_states WHERE chat_id = ?), 0),
                    CURRENT_TIMESTAMP
                )
            """, (chat_id, persona.value, chat_id))
            
            logger.info("Chat persona updated", 
                       chat_id=chat_id, 
                       persona=persona.value)
    
    def _update_message_count(self, conn: sqlite3.Connection, chat_id: int) -> None:
        """Обновляет счетчик сообщений в состоянии чата"""
        conn.execute("""
            INSERT OR REPLACE INTO chat_states 
            (chat_id, current_persona, message_count, last_activity)
            VALUES (
                ?,
                COALESCE((SELECT current_persona FROM chat_states WHERE chat_id = ?), 'business'),
                (SELECT COUNT(*) FROM messages WHERE chat_id = ?),
                CURRENT_TIMESTAMP
            )
        """, (chat_id, chat_id, chat_id))
    
    def _cleanup_old_messages(self, conn: sqlite3.Connection, chat_id: int) -> None:
        """Удаляет старые сообщения если превышен лимит"""
        conn.execute("""
            DELETE FROM messages 
            WHERE chat_id = ? 
            AND id NOT IN (
                SELECT id FROM messages 
                WHERE chat_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            )
        """, (chat_id, chat_id, config.MESSAGE_HISTORY_LIMIT))


# Глобальный экземпляр менеджера базы данных
db_manager = DatabaseManager()
