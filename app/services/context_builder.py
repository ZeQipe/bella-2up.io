"""
Сервис для формирования контекста сообщений перед отправкой в DeepSeek
Реализует логику дублирования системного промта при большой истории
"""
from typing import List, Dict, Optional
import structlog

from ..config import PersonaType
from ..models import Message, MessageRole
from .prompt_service import prompt_service

logger = structlog.get_logger()


class ContextBuilder:
    """Сервис для построения финального контекста сообщений"""
    
    def __init__(self):
        self.system_prompt_threshold = 5  # Порог для дублирования системного промта
    
    def build_context_for_ai(
        self,
        messages: List[Message],
        persona: PersonaType,
        current_message_content: str,
        promotions_info: Optional[str] = None,
        vector_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Формирует финальный контекст для отправки в DeepSeek API
        
        Логика:
        - Если сообщений <= 5: system_prompt -> history -> current_message
        - Если сообщений > 5: system_prompt -> history -> system_prompt -> current_message
        
        Args:
            messages: История сообщений (последние 20 за час)
            persona: Активная персона
            current_message_content: Текущее сообщение пользователя
            promotions_info: Информация о промоакциях
            vector_context: Контекст из векторного поиска
            
        Returns:
            Список сообщений в формате для DeepSeek API
        """
        try:
            # Получаем системный промт с подстановками
            system_prompt = prompt_service.get_system_prompt(
                persona=persona,
                promotions_info=promotions_info,
                context_info=vector_context
            )
            
            # Начинаем с системного промта
            context_messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Добавляем историю сообщений (исключаем системные сообщения)
            history_messages = [
                msg for msg in messages 
                if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]
            ]
            
            # Конвертируем историю в формат API
            for message in history_messages:
                context_messages.append(message.to_dict())
            
            # Проверяем нужно ли дублировать системный промт
            if len(history_messages) > self.system_prompt_threshold:
                logger.debug("Adding duplicate system prompt", 
                           history_count=len(history_messages),
                           threshold=self.system_prompt_threshold)
                
                # Добавляем системный промт повторно
                context_messages.append({
                    "role": "system", 
                    "content": system_prompt
                })
            
            # Добавляем текущее сообщение пользователя
            context_messages.append({
                "role": "user",
                "content": current_message_content
            })
            
            logger.debug("Context built for AI",
                        persona=persona.value,
                        total_messages=len(context_messages),
                        history_messages=len(history_messages),
                        has_duplicate_system=len(history_messages) > self.system_prompt_threshold,
                        has_promotions=bool(promotions_info),
                        has_vector_context=bool(vector_context))
            
            return context_messages
            
        except Exception as e:
            logger.error("Failed to build context for AI", 
                        persona=persona.value,
                        error=str(e))
            
            # Fallback: минимальный контекст
            fallback_prompt = f"Ты консультант казино в режиме {persona.value}. Отвечай профессионально."
            return [
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": current_message_content}
            ]
    
    def get_context_statistics(self, context_messages: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Возвращает статистику по контексту
        
        Args:
            context_messages: Контекст сообщений
            
        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_messages": len(context_messages),
            "system_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "total_tokens_estimate": 0
        }
        
        for message in context_messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                stats["system_messages"] += 1
            elif role == "user":
                stats["user_messages"] += 1
            elif role == "assistant":
                stats["assistant_messages"] += 1
            
            # Примерная оценка токенов (1 токен ≈ 4 символа для русского)
            stats["total_tokens_estimate"] += len(content) // 4
        
        return stats


class ContextBuilderError(Exception):
    """Исключение для ошибок построения контекста"""
    pass


# Глобальный экземпляр построителя контекста
context_builder = ContextBuilder()
