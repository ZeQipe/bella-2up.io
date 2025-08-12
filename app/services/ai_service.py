"""
Сервис для работы с AI API (DeepSeek)
Реализует интеграцию с DeepSeek Chat API для генерации ответов
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json as _json
import httpx
import structlog

from ..config import config, PersonaType
from ..models import Message
from ..database import db_manager
from .context_builder import context_builder
from ..services.vector_service import vector_service
from .query_translator import query_translator
from ..config import config
import os


def _ensure_audit_dir(path: str):
    audit_dir = os.path.dirname(path)
    if audit_dir and not os.path.exists(audit_dir):
        os.makedirs(audit_dir, exist_ok=True)


def _append_audit_log(lines: list[str], path: str) -> None:
    try:
        _ensure_audit_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        # Не мешаем основному потоку при ошибке логирования в файл
        pass

logger = structlog.get_logger()


class AIServiceInterface(ABC):
    """Абстрактный интерфейс для AI сервиса"""
    
    @abstractmethod
    async def generate_response(
        self, 
        chat_id: int,
        current_message: str,
        persona: PersonaType,
        promotions_info: Optional[str] = None
    ) -> str:
        """Генерирует ответ на основе истории сообщений и персоны"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Проверяет доступность AI сервиса"""
        pass


class DeepSeekService(AIServiceInterface):
    """Сервис для работы с DeepSeek Chat API"""
    
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.api_base = config.DEEPSEEK_API_BASE
        self.model = config.DEEPSEEK_MODEL
    
    async def generate_response(
        self, 
        chat_id: int,
        current_message: str,
        persona: PersonaType,
        promotions_info: Optional[str] = None
    ) -> str:
        """Генерирует ответ через DeepSeek Chat API с учётом истории и контекста."""
        try:
            # Получаем историю сообщений за последний час (максимум 20)
            messages = db_manager.get_chat_history(chat_id, limit=20, hours_back=1)
            
            # Получаем векторный контекст для текущего сообщения
            vector_context = await self._get_vector_context(current_message)
            
            # Формируем финальный контекст для AI
            context_messages = context_builder.build_context_for_ai(
                messages=messages,
                persona=persona,
                current_message_content=current_message,
                promotions_info=promotions_info,
                vector_context=vector_context
            )
            
            # Аудит: сохраняем отправку в DeepSeek (контекст запроса)
            from datetime import datetime as _dt
            timestamp = _dt.now().isoformat()
            role_map = {"user": "пользователь", "assistant": "ассистент", "system": "система"}
            audit_lines = [
                f"=== DEEPSEEK REQUEST @ {timestamp} ===",
                f"chat_id={chat_id} persona={persona.value}",
            ]
            for msg in context_messages:
                role = role_map.get(msg.get("role", ""), msg.get("role", ""))
                content = msg.get("content", "")
                # ограничим длину для строкового аудита, но сохраним полный промт ниже
                audit_lines.append(f"[{role}] {content[:300].replace('\n', ' ')}{'...' if len(content)>300 else ''}")
            # Полный промт (весь массив) одной строкой
            audit_lines.append("--- FULL PROMPT (JSON-like dump) ---")
            try:
                audit_lines.append(_json.dumps(context_messages, ensure_ascii=False)[:5000])
            except Exception:
                audit_lines.append("[unserializable prompt]")
            _append_audit_log(audit_lines, config.AUDIT_DEEPSEEK_LOG_PATH)
            
            # Вызов DeepSeek Chat API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": context_messages,
                "stream": False,
            }
            async with httpx.AsyncClient(base_url=self.api_base, timeout=30.0) as client:
                resp = await client.post("/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            
            # Извлекаем ответ
            response_text = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
            )
            if not response_text:
                raise ValueError("Empty response from DeepSeek API")

            # Аудит: записываем ответ
            _append_audit_log([
                f"--- DEEPSEEK RESPONSE ---",
                response_text,
                f"=== END DEEPSEEK @ {timestamp} ===",
                ""
            ], config.AUDIT_DEEPSEEK_LOG_PATH)
            return response_text
                
        except Exception as e:
            logger.error("AI service error", chat_id=chat_id, error=str(e))
            # Аудит: ошибка
            _append_audit_log([
                f"--- DEEPSEEK ERROR ---",
                f"error={str(e)}",
            ], config.AUDIT_DEEPSEEK_LOG_PATH)
            # Возвращаем фолбэк согласно ТЗ
            return config.ERROR_MESSAGE
    
    async def _get_vector_context(self, query: str) -> Optional[str]:
        """Получает контекст из векторного поиска с переводом запроса"""
        try:
            logger.info("🔍 Starting vector context search", original_query=query[:100])
            
            # Переводим запрос на английский для лучшего поиска
            translated_query = await query_translator.translate_query(query)
            
            if translated_query is None:
                logger.info("💬 Query identified as casual chat, skipping vector search")
                return None
            
            if translated_query != query:
                logger.info("🌐 Query translated for search",
                           original=query[:50],
                           translated=translated_query[:50])
            
            # Выполняем векторный поиск с переведенным запросом
            search_results = await vector_service.search_similar(translated_query)
            
            if not search_results:
                logger.info("📭 No relevant documents found, continuing without context")
                # Аудит: логируем попытку векторного поиска
                _append_audit_log([
                    "=== VECTOR SEARCH ===",
                    f"original_query={query[:200]}",
                    f"translated_query={translated_query[:200]}",
                    "results=0",
                    "=== END VECTOR SEARCH ===",
                    ""
                ], config.AUDIT_VECTOR_LOG_PATH)
                return None
            
            # Формируем контекст из найденных документов
            context_parts = []
            for result in search_results:
                content = result.get("content", "")
                similarity = result.get("similarity", 0.0)
                
                if content:
                    context_parts.append(f"[Релевантность: {similarity:.2f}] {content}")
            
            final_context = "\n\n".join(context_parts) if context_parts else None

            # Аудит: логируем найденные векторы
            lines = [
                "=== VECTOR SEARCH ===",
                f"original_query={query[:200]}",
                f"translated_query={translated_query[:200]}",
                f"results={len(search_results)}",
            ]
            for i, r in enumerate(search_results[:5]):
                lines.append(f"[{i+1}] score={r.get('similarity',0):.3f} file={r.get('metadata',{}).get('file','unknown')} content='{r.get('content','')[:200]}...'")
            lines.append("=== END VECTOR SEARCH ===")
            lines.append("")
            _append_audit_log(lines, config.AUDIT_VECTOR_LOG_PATH)
            
            if final_context:
                logger.info("✅ Vector context prepared",
                           documents_count=len(search_results),
                           context_length=len(final_context))
            
            return final_context
            
        except Exception as e:
            logger.warning("⚠️ Failed to get vector context, continuing without it",
                          query=query[:50], error=str(e))
            return None
    
    async def health_check(self) -> bool:
        """Проверяет доступность DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            async with httpx.AsyncClient(base_url=self.api_base, timeout=10.0) as client:
                resp = await client.post("/v1/chat/completions", headers=headers, json=payload)
                return resp.status_code == 200
        except Exception:
            return False
    



class AIServiceError(Exception):
    """Исключение для ошибок AI сервиса"""
    pass


def get_ai_service() -> AIServiceInterface:
    """Фабрика для получения экземпляра AI сервиса"""
    return DeepSeekService()


# Глобальный экземпляр AI сервиса
ai_service = get_ai_service()
