"""
Сервис для создания эмбеддингов через OpenAI API
Реализует создание векторных представлений текста
"""
import os
import hashlib
from typing import List, Optional
import structlog
import openai
from openai import OpenAI

from ..config import config

logger = structlog.get_logger()


class EmbeddingService:
    """Сервис для создания эмбеддингов через OpenAI API"""
    
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_EMBEDDING_MODEL
        self.client: Optional[OpenAI] = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для текста через OpenAI API
        
        Args:
            text: Текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
            
        Raises:
            EmbeddingServiceError: При ошибке создания эмбеддинга
        """
        if not self.client:
            raise EmbeddingServiceError("OpenAI client not initialized. Check API key.")
        
        if not text.strip():
            raise EmbeddingServiceError("Text cannot be empty")
        
        try:
            # Очищаем текст от лишних пробелов и переносов
            clean_text = " ".join(text.strip().split())
            
            # Создаем эмбеддинг через OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            logger.debug("Embedding created", 
                        text_length=len(clean_text),
                        embedding_dimension=len(embedding),
                        model=self.model)
            
            return embedding
            
        except openai.APIError as e:
            logger.error("OpenAI API error", error=str(e))
            raise EmbeddingServiceError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error("Unexpected error creating embedding", error=str(e))
            raise EmbeddingServiceError(f"Failed to create embedding: {e}")
    
    def create_embeddings_batch_sync(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов пакетами
        
        Args:
            texts: Список текстов
            batch_size: Размер пакета для обработки
            
        Returns:
            Список векторов эмбеддингов
        """
        if not self.client:
            raise EmbeddingServiceError("OpenAI client not initialized. Check API key.")
        
        if not texts:
            return []
        
        embeddings = []
        
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
            
            # Обрабатываем тексты пакетами
            for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
                batch = texts[i:i + batch_size]
                
                # Очищаем тексты
                clean_batch = [" ".join(text.strip().split()) for text in batch if text.strip()]
                
                if not clean_batch:
                    continue
                
                logger.info(f"Creating embeddings for batch {batch_num}/{total_batches} ({len(clean_batch)} texts)")
                
                # Создаем эмбеддинги для пакета
                response = self.client.embeddings.create(
                    model=self.model,
                    input=clean_batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"✅ Batch {batch_num}/{total_batches} completed. Total processed: {len(embeddings)}")
            
            logger.info("All embeddings created",
                       total_texts=len(texts),
                       total_embeddings=len(embeddings),
                       model=self.model)
            
            return embeddings
            
        except openai.APIError as e:
            logger.error("OpenAI API error in batch processing", error=str(e))
            raise EmbeddingServiceError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error("Unexpected error in batch processing", error=str(e))
            raise EmbeddingServiceError(f"Failed to create batch embeddings: {e}")
    
    def get_text_hash(self, text: str) -> str:
        """
        Создает хеш для текста для проверки изменений
        
        Args:
            text: Исходный текст
            
        Returns:
            SHA-256 хеш текста
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    async def health_check(self) -> bool:
        """
        Проверяет доступность OpenAI API
        
        Returns:
            True если API доступен, False иначе
        """
        if not self.client:
            return False
        
        try:
            # Тестовый запрос с коротким текстом
            test_response = self.client.embeddings.create(
                model=self.model,
                input="test",
                encoding_format="float"
            )
            
            return len(test_response.data) > 0 and len(test_response.data[0].embedding) > 0
            
        except Exception as e:
            logger.warning("OpenAI API health check failed", error=str(e))
            return False


class EmbeddingServiceError(Exception):
    """Исключение для ошибок сервиса эмбеддингов"""
    pass


# Глобальный экземпляр сервиса эмбеддингов
embedding_service = EmbeddingService()
