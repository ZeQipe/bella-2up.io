"""
Сервис для перевода пользовательских запросов на английский язык
Использует DeepSeek для нормализации запросов перед векторным поиском
"""
from typing import Optional
import structlog
import httpx
import json

from ..config import config

logger = structlog.get_logger()


class QueryTranslator:
    """Сервис для перевода запросов на английский язык через DeepSeek"""
    
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.model = config.DEEPSEEK_MODEL
        
        # Промт для перевода и нормализации запросов
        self.translation_prompt = """You are a query translator for a casino support system. Your task is to translate user queries to English for vector search purposes.

Rules:
1. Translate the query to English while preserving the meaning
2. Keep casino-specific terms (deposit, withdrawal, bonus, registration, etc.)
3. If the query is casual chat/flirting with no casino context, return "CASUAL_CHAT"
4. If already in English, improve the query for better search results
5. Keep the translation concise and search-friendly
6. Focus on extracting the main question/intent

Examples:
- "как зарегистрироваться?" → "how to register account"
- "привет красавчик, как дела?" → "CASUAL_CHAT"  
- "минимальный депозит" → "minimum deposit amount"
- "где мои бонусы?" → "where are my bonuses"
- "how to withdraw money" → "how to withdraw money"

Translate this query:"""
    
    async def translate_query(self, original_query: str) -> Optional[str]:
        """
        Переводит запрос пользователя на английский для векторного поиска
        
        Args:
            original_query: Оригинальный запрос пользователя
            
        Returns:
            Переведенный запрос или None если это обычная болтовня
        """
        if not original_query or not original_query.strip():
            logger.debug("Empty query provided for translation")
            return None
        
        try:
            # Вызов DeepSeek API для перевода
            translated_query = await self._call_deepseek_translation(original_query)
            
            if translated_query == "CASUAL_CHAT":
                logger.info("Query identified as casual chat", 
                           original_query=original_query[:50])
                return None
            
            logger.info("Query translated for vector search",
                       original=original_query[:50],
                       translated=translated_query[:50])
            
            return translated_query
            
        except Exception as e:
            logger.warning("Failed to translate query, using original",
                          original_query=original_query[:50],
                          error=str(e))
            # Fallback: возвращаем оригинальный запрос
            return original_query
    
    async def _call_deepseek_translation(self, query: str) -> str:
        """Вызывает DeepSeek API для перевода запроса"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = [
            {"role": "system", "content": self.translation_prompt},
            {"role": "user", "content": query}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        async with httpx.AsyncClient(base_url=config.DEEPSEEK_API_BASE, timeout=15.0) as client:
            resp = await client.post("/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        
        translated = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
        )
        
        if not translated:
            raise ValueError("Empty translation response from DeepSeek")
            
        return translated
    
    async def health_check(self) -> bool:
        """Проверяет доступность сервиса перевода"""
        try:
            # Простой тест перевода
            test_result = await self.translate_query("тест")
            return test_result is not None
        except Exception:
            return False


class QueryTranslatorError(Exception):
    """Исключение для ошибок сервиса перевода запросов"""
    pass


# Глобальный экземпляр переводчика запросов
query_translator = QueryTranslator()
