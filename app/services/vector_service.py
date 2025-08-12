"""
Сервис для работы с векторной базой данных (ChromaDB)
Реализует создание, сохранение и поиск векторов
"""
import os
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import structlog
import chromadb
from chromadb.config import Settings

from ..config import config
from .embedding_service import embedding_service, EmbeddingServiceError

logger = structlog.get_logger()


class VectorServiceInterface(ABC):
    """Абстрактный интерфейс для векторного сервиса"""
    
    @abstractmethod
    def initialize_knowledge_base_sync(self, kb_path: str) -> None:
        """Инициализирует векторную базу знаний из файлов"""
        pass
    
    @abstractmethod
    async def search_similar(self, query: str, limit: int = None, threshold: float = None) -> List[Dict]:
        """Ищет похожие документы по запросу"""
        pass
    
    @abstractmethod
    async def add_document(self, content: str, metadata: Dict) -> str:
        """Добавляет документ в векторную базу"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Проверяет доступность векторного сервиса"""
        pass


class ChromaVectorService(VectorServiceInterface):
    """Сервис для работы с ChromaDB"""
    
    def __init__(self):
        self.db_path = config.CHROMA_DB_PATH
        self.collection_name = "casino_knowledge_base"
        self.embedding_model = config.OPENAI_EMBEDDING_MODEL
        
        # Инициализация ChromaDB
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialized = False
        
        # Кеш для отслеживания изменений в файлах
        self._file_hashes: Dict[str, str] = {}
        
    def _ensure_directory(self) -> None:
        """Создает директорию для ChromaDB если не существует"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
            logger.info("Created ChromaDB directory", path=self.db_path)
    
    def _initialize_client(self) -> None:
        """Инициализирует клиент ChromaDB"""
        if self.client is not None:
            return
            
        try:
            self._ensure_directory()
            
            # Создаем клиент ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Получаем или создаем коллекцию
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info("Connected to existing ChromaDB collection", 
                           collection=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Casino knowledge base vectors"}
                )
                logger.info("Created new ChromaDB collection", 
                           collection=self.collection_name)
            
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize ChromaDB client", error=str(e))
            raise VectorServiceError(f"ChromaDB initialization failed: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Вычисляет хеш файла для отслеживания изменений"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning("Failed to compute file hash", file=file_path, error=str(e))
            return ""
    
    def _should_rebuild_vectors(self, kb_path: str) -> bool:
        """Проверяет нужно ли пересоздавать векторы"""
        if not config.VECTOR_REBUILD_ON_CHANGES:
            return False
            
        if not os.path.exists(kb_path):
            return False
        
        # Проверяем изменения в txt файлах
        current_hashes = {}
        for root, dirs, files in os.walk(kb_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    current_hashes[file_path] = self._get_file_hash(file_path)
        
        # Сравниваем с кешированными хешами
        if current_hashes != self._file_hashes:
            self._file_hashes = current_hashes
            logger.info("File changes detected, vectors will be rebuilt")
            return True
        
        return False
    
    def initialize_knowledge_base_sync(self, kb_path: str) -> None:
        """
        Инициализация векторной базы знаний из txt файлов (СИНХРОННО)
        1. Читает файлы из kb_path
        2. Разбивает на строки
        3. Создает эмбеддинги через OpenAI API
        4. Сохраняет в ChromaDB
        """
        try:
            logger.info("Initializing ChromaDB client")
            self._initialize_client()
            logger.info("ChromaDB client initialized successfully")
            
            if not os.path.exists(kb_path):
                logger.warning("Knowledge base path does not exist", path=kb_path)
                return
            
            # Проверяем нужно ли пересоздавать векторы
            if not self._should_rebuild_vectors(kb_path) and self.collection.count() > 0:
                logger.info("Vectors are up to date, skipping rebuild")
                return
            
            logger.info("Initializing knowledge base", kb_path=kb_path)
            
            # Очищаем существующую коллекцию
            if self.collection.count() > 0:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Casino knowledge base vectors"}
                )
                logger.info("Cleared existing vectors")
            
            # Собираем все txt файлы
            documents = []
            metadatas = []
            ids = []
            doc_counter = 0
            
            for root, dirs, files in os.walk(kb_path):
                for file in files:
                    if not file.endswith('.txt'):
                        continue
                    # Исключаем файл промоакций, если размещен в kb/
                    if file.lower() == os.path.basename(config.PROMOTIONS_FILE).lower():
                        logger.debug("Skipping promotions file from embeddings", file=file)
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Разбиваем на строки и фильтруем пустые
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        
                        for line_num, line in enumerate(lines):
                            if len(line) < 10:  # Пропускаем слишком короткие строки
                                continue
                                
                            doc_counter += 1
                            documents.append(line)
                            ids.append(f"doc_{doc_counter}")
                            metadatas.append({
                                "file": file,
                                "file_path": file_path,
                                "line_number": line_num + 1,
                                "created_at": datetime.now().isoformat()
                            })
                            
                    except Exception as e:
                        logger.warning("Failed to read file", file=file_path, error=str(e))
                        continue
            
            if not documents:
                logger.warning("No documents found in knowledge base")
                return
            
            logger.info("Processing documents", total_documents=len(documents))
            
            # Создаем эмбеддинги пакетами
            try:
                logger.info("Creating embeddings via OpenAI API", document_count=len(documents))
                embeddings = embedding_service.create_embeddings_batch_sync(documents)
                logger.info("Embeddings created successfully", embedding_count=len(embeddings))
                
                if len(embeddings) != len(documents):
                    raise VectorServiceError("Mismatch between documents and embeddings count")
                
                # Сохраняем в ChromaDB
                logger.info("Saving embeddings to ChromaDB", vector_count=len(embeddings))
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info("Embeddings saved to ChromaDB successfully")
                
                logger.info("Knowledge base initialized successfully",
                           total_documents=len(documents),
                           total_embeddings=len(embeddings))
                
            except EmbeddingServiceError as e:
                logger.error("Failed to create embeddings", error=str(e))
                raise VectorServiceError(f"Embedding creation failed: {e}")
            
        except Exception as e:
            logger.error("Failed to initialize knowledge base", error=str(e))
            raise VectorServiceError(f"Knowledge base initialization failed: {e}")
    
    async def search_similar(self, query: str, limit: int = None, threshold: float = None) -> List[Dict]:
        """
        Поиск похожих документов по запросу
        1. Создает эмбеддинг для запроса
        2. Ищет похожие в ChromaDB
        3. Возвращает отранжированные результаты
        """
        try:
            self._initialize_client()
            
            if not query.strip():
                return []
            
            # Используем конфигурационные значения по умолчанию
            search_limit = limit or config.VECTOR_SEARCH_LIMIT
            similarity_threshold = threshold or config.VECTOR_SIMILARITY_THRESHOLD
            
            # Проверяем есть ли документы в коллекции
            if self.collection.count() == 0:
                logger.warning("No documents in vector database")
                return []
            
            # Создаем эмбеддинг для запроса
            query_embedding = await embedding_service.create_embedding(query)
            
            # Выполняем поиск в ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Обрабатываем результаты
            results = []
            if search_results["documents"] and search_results["documents"][0]:
                documents = search_results["documents"][0]
                metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
                distances = search_results["distances"][0] if search_results["distances"] else []
                
                for i, document in enumerate(documents):
                    # ChromaDB возвращает расстояние (чем меньше, тем лучше)
                    # Преобразуем в схожесть (чем больше, тем лучше)
                    distance = distances[i] if i < len(distances) else 1.0
                    # Нормализуем расстояние в [0,1] с учетом метрики cosine (обычно [0,2])
                    if distance <= 1.0:
                        # Уже в разумном диапазоне [0,1]
                        similarity = 1.0 - distance
                    elif distance <= 2.0:
                        # Косинусная "distance" может быть в [0,2]; нормализуем
                        similarity = 1.0 - (distance / 2.0)
                    else:
                        # Нестандартные значения — даем очень низкую схожесть
                        similarity = max(0.0, 1.0 / (1.0 + distance))
                    
                    # Фильтруем по порогу схожести
                    if similarity < similarity_threshold:
                        continue
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    results.append({
                        "content": document,
                        "similarity": similarity,
                        "distance": distance,
                        "metadata": metadata
                    })
            
            # Сортируем по схожести (убывание)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Детальное логирование результатов поиска
            logger.info("=== VECTOR SEARCH RESULTS ===",
                       query=query[:100],
                       results_found=len(results),
                       threshold=similarity_threshold,
                       limit=search_limit)
            
            if results:
                logger.info("📄 Found documents:")
                for i, result in enumerate(results[:3]):  # Показываем первые 3
                    content = result.get("content", "")[:100]
                    similarity = result.get("similarity", 0.0)
                    metadata = result.get("metadata", {})
                    file_name = metadata.get("file", "unknown")
                    
                    logger.info(f"  [{i+1}] Score: {similarity:.3f} | File: {file_name} | Distance: {result.get('distance', 0):.3f}")
                    logger.info(f"      Content: '{content}...'")
                
                if len(results) > 3:
                    logger.info(f"  ... and {len(results) - 3} more documents")
            else:
                logger.warning("❌ No documents found above similarity threshold",
                             threshold=similarity_threshold)
            
            logger.info("=== END VECTOR SEARCH ===")
            
            logger.debug("Vector search completed",
                        query=query[:50],
                        results_count=len(results),
                        threshold=similarity_threshold,
                        limit=search_limit)
            
            return results
            
        except EmbeddingServiceError as e:
            logger.error("Failed to create query embedding", query=query[:50], error=str(e))
            raise VectorServiceError(f"Query embedding failed: {e}")
        except Exception as e:
            logger.error("Vector search failed", query=query[:50], error=str(e))
            raise VectorServiceError(f"Vector search failed: {e}")
    
    async def add_document(self, content: str, metadata: Dict) -> str:
        """
        Добавление документа в векторную базу
        Создает эмбеддинг и сохраняет в ChromaDB
        """
        try:
            self._initialize_client()
            
            if not content.strip():
                raise VectorServiceError("Document content cannot be empty")
            
            # Генерируем уникальный ID для документа
            doc_id = f"doc_{int(datetime.now().timestamp() * 1000000)}"
            
            # Создаем эмбеддинг для документа
            embedding = await embedding_service.create_embedding(content)
            
            # Добавляем метаданные
            full_metadata = {
                **metadata,
                "added_at": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            # Сохраняем в ChromaDB
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[full_metadata],
                ids=[doc_id]
            )
            
            logger.debug("Document added to vector database",
                        doc_id=doc_id,
                        content_length=len(content))
            
            return doc_id
            
        except EmbeddingServiceError as e:
            logger.error("Failed to create embedding for document", error=str(e))
            raise VectorServiceError(f"Document embedding failed: {e}")
        except Exception as e:
            logger.error("Failed to add document", error=str(e))
            raise VectorServiceError(f"Failed to add document: {e}")
    
    async def health_check(self) -> bool:
        """
        Проверяет доступность векторного сервиса
        
        Returns:
            True если сервис доступен, False иначе
        """
        try:
            self._initialize_client()
            
            # Проверяем доступность ChromaDB
            if not self.client or not self.collection:
                return False
            
            # Проверяем доступность сервиса эмбеддингов
            embedding_health = await embedding_service.health_check()
            
            return embedding_health
            
        except Exception as e:
            logger.warning("Vector service health check failed", error=str(e))
            return False


class VectorServiceError(Exception):
    """Исключение для ошибок векторного сервиса"""
    pass


def get_vector_service() -> VectorServiceInterface:
    """Фабрика для получения экземпляра векторного сервиса"""
    return ChromaVectorService()


# Глобальный экземпляр векторного сервиса
vector_service = get_vector_service()
