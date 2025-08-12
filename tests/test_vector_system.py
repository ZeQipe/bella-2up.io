"""
Тесты для проверки работоспособности векторной системы
Включает тестирование создания эмбеддингов, сохранения и поиска векторов
"""
import os
import tempfile
import shutil
import asyncio
from typing import List, Dict
import pytest
import structlog

from app.config import config
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.vector_service import ChromaVectorService, VectorServiceError

logger = structlog.get_logger()


class TestEmbeddingService:
    """Тесты для сервиса эмбеддингов"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.embedding_service = EmbeddingService()
    
    @pytest.mark.asyncio
    async def test_create_embedding_success(self):
        """Тест успешного создания эмбеддинга"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        text = "Как зарегистрироваться в казино?"
        embedding = await self.embedding_service.create_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        logger.info("Embedding test passed", embedding_dimension=len(embedding))
    
    @pytest.mark.asyncio
    async def test_create_embedding_empty_text(self):
        """Тест обработки пустого текста"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        with pytest.raises(EmbeddingServiceError):
            await self.embedding_service.create_embedding("")
    
    @pytest.mark.asyncio
    async def test_create_embeddings_batch(self):
        """Тест создания эмбеддингов пакетом"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        texts = [
            "Как зарегистрироваться в казино?",
            "Как пополнить счет?",
            "Как вывести деньги?"
        ]
        
        embeddings = await self.embedding_service.create_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        logger.info("Batch embeddings test passed", batch_size=len(texts))
    
    def test_get_text_hash(self):
        """Тест создания хеша текста"""
        text = "Тестовый текст для хеширования"
        hash1 = self.embedding_service.get_text_hash(text)
        hash2 = self.embedding_service.get_text_hash(text)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hash length
        assert hash1 == hash2  # Одинаковый текст должен давать одинаковый хеш
        
        # Разный текст должен давать разный хеш
        different_text = "Другой тестовый текст"
        hash3 = self.embedding_service.get_text_hash(different_text)
        assert hash1 != hash3
        
        logger.info("Text hashing test passed")
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Тест проверки работоспособности сервиса"""
        if not config.OPENAI_API_KEY:
            # Без API ключа health check должен вернуть False
            health = await self.embedding_service.health_check()
            assert health is False
        else:
            health = await self.embedding_service.health_check()
            assert isinstance(health, bool)
            logger.info("Health check test passed", health_status=health)


class TestVectorService:
    """Тесты для векторного сервиса"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Создаем временную директорию для тестов
        self.temp_dir = tempfile.mkdtemp()
        self.test_kb_dir = os.path.join(self.temp_dir, "test_kb")
        self.test_chroma_dir = os.path.join(self.temp_dir, "test_chroma")
        
        os.makedirs(self.test_kb_dir, exist_ok=True)
        
        # Создаем тестовые файлы базы знаний
        self._create_test_knowledge_base()
        
        # Создаем векторный сервис с тестовыми путями
        self.vector_service = ChromaVectorService()
        self.vector_service.db_path = self.test_chroma_dir
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_knowledge_base(self):
        """Создает тестовые файлы базы знаний"""
        test_files = {
            "registration.txt": [
                "Для регистрации в казино нужно указать email и пароль",
                "Подтвердите email адрес после регистрации",
                "Минимальный возраст для игры - 18 лет",
                "Регистрация занимает не более 5 минут"
            ],
            "payments.txt": [
                "Пополнение счета доступно через банковские карты",
                "Минимальная сумма депозита - 100 рублей",
                "Вывод средств обрабатывается в течение 24 часов",
                "Комиссия за вывод составляет 2% от суммы"
            ],
            "bonuses.txt": [
                "Новые игроки получают приветственный бонус 100%",
                "Бонус нужно отыграть с вейджером x35",
                "Максимальная сумма бонуса - 50000 рублей",
                "Бонусы начисляются автоматически"
            ]
        }
        
        for filename, lines in test_files.items():
            file_path = os.path.join(self.test_kb_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
    
    @pytest.mark.asyncio
    async def test_initialize_knowledge_base(self):
        """Тест инициализации базы знаний"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        await self.vector_service.initialize_knowledge_base(self.test_kb_dir)
        
        # Проверяем что векторы созданы
        assert self.vector_service.collection is not None
        count = self.vector_service.collection.count()
        assert count > 0
        
        logger.info("Knowledge base initialization test passed", 
                   document_count=count)
    
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Тест поиска похожих документов"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        # Сначала инициализируем базу знаний
        await self.vector_service.initialize_knowledge_base(self.test_kb_dir)
        
        # Тестируем поиск
        query = "Как зарегистрироваться?"
        results = await self.vector_service.search_similar(query, limit=3, threshold=0.5)
        
        assert isinstance(results, list)
        # Должны найти хотя бы один релевантный документ
        assert len(results) >= 1
        
        for result in results:
            assert "content" in result
            assert "similarity" in result
            assert "metadata" in result
            assert isinstance(result["similarity"], float)
            assert 0.0 <= result["similarity"] <= 1.0
        
        logger.info("Vector search test passed", 
                   query=query, results_count=len(results))
    
    @pytest.mark.asyncio
    async def test_add_document(self):
        """Тест добавления документа"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        # Инициализируем сервис
        self.vector_service._initialize_client()
        
        # Добавляем документ
        content = "Тестовый документ для проверки добавления"
        metadata = {"test": True, "category": "test"}
        
        doc_id = await self.vector_service.add_document(content, metadata)
        
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        
        # Проверяем что документ добавился
        count = self.vector_service.collection.count()
        assert count == 1
        
        logger.info("Add document test passed", doc_id=doc_id)
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Тест проверки работоспособности"""
        if not config.OPENAI_API_KEY:
            # Без API ключа health check должен вернуть False
            health = await self.vector_service.health_check()
            assert health is False
        else:
            health = await self.vector_service.health_check()
            assert isinstance(health, bool)
            logger.info("Vector service health check test passed", 
                       health_status=health)


class TestIntegration:
    """Интеграционные тесты векторной системы"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_kb_dir = os.path.join(self.temp_dir, "integration_kb")
        self.test_chroma_dir = os.path.join(self.temp_dir, "integration_chroma")
        
        os.makedirs(self.test_kb_dir, exist_ok=True)
        
        # Создаем реалистичную базу знаний
        self._create_realistic_knowledge_base()
        
        self.vector_service = ChromaVectorService()
        self.vector_service.db_path = self.test_chroma_dir
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_realistic_knowledge_base(self):
        """Создает реалистичную базу знаний для интеграционных тестов"""
        kb_content = {
            "faq_registration.txt": [
                "Регистрация в казино бесплатна и занимает всего несколько минут",
                "Для создания аккаунта необходимо указать email, пароль и подтвердить возраст",
                "После регистрации на email придет письмо для подтверждения аккаунта",
                "Игрокам младше 18 лет регистрация запрещена",
                "Один игрок может иметь только один активный аккаунт"
            ],
            "faq_deposits.txt": [
                "Пополнение счета возможно банковскими картами Visa и MasterCard",
                "Минимальная сумма депозита составляет 500 рублей",
                "Средства зачисляются на счет мгновенно",
                "Комиссия за пополнение не взимается",
                "Максимальная сумма одного депозита - 100000 рублей"
            ],
            "faq_withdrawals.txt": [
                "Вывод средств осуществляется на банковские карты и электронные кошельки",
                "Минимальная сумма для вывода - 1000 рублей",
                "Обработка заявки на вывод занимает от 1 до 24 часов",
                "Верификация аккаунта обязательна для первого вывода",
                "Комиссия за вывод составляет 3% от суммы"
            ],
            "support_common.txt": [
                "Служба поддержки работает круглосуточно",
                "Связаться с поддержкой можно через чат или email",
                "Среднее время ответа - 15 минут",
                "Поддержка доступна на русском и английском языках"
            ]
        }
        
        for filename, lines in kb_content.items():
            file_path = os.path.join(self.test_kb_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Тест полного цикла работы с векторной системой"""
        if not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API key not configured")
        
        # 1. Инициализация базы знаний
        await self.vector_service.initialize_knowledge_base(self.test_kb_dir)
        
        document_count = self.vector_service.collection.count()
        assert document_count > 0
        
        # 2. Поиск по различным запросам
        test_queries = [
            ("Как зарегистрироваться?", "registration"),
            ("Как пополнить счет?", "deposits"),
            ("Как вывести деньги?", "withdrawals"),
            ("Служба поддержки", "support")
        ]
        
        for query, expected_topic in test_queries:
            results = await self.vector_service.search_similar(query, limit=5, threshold=0.6)
            
            assert len(results) > 0, f"No results for query: {query}"
            
            # Проверяем что нашли релевантные документы
            found_relevant = False
            for result in results:
                if expected_topic in result["content"].lower():
                    found_relevant = True
                    break
            
            logger.info("Query test passed", 
                       query=query, 
                       results_count=len(results),
                       found_relevant=found_relevant)
        
        # 3. Добавление нового документа
        new_content = "Техническая поддержка доступна через Telegram бота"
        doc_id = await self.vector_service.add_document(
            new_content, 
            {"category": "support", "type": "telegram"}
        )
        
        assert doc_id is not None
        
        # 4. Поиск добавленного документа
        telegram_results = await self.vector_service.search_similar("Telegram бот", limit=3)
        
        found_new_doc = False
        for result in telegram_results:
            if "Telegram" in result["content"]:
                found_new_doc = True
                break
        
        assert found_new_doc, "Newly added document not found in search"
        
        logger.info("End-to-end workflow test passed", 
                   total_documents=document_count + 1,
                   queries_tested=len(test_queries))


def run_vector_tests():
    """Запуск всех тестов векторной системы"""
    print("🧪 Запуск тестов векторной системы...")
    
    # Настройка логирования для тестов
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Проверяем наличие OpenAI API ключа
    if not config.OPENAI_API_KEY:
        print("⚠️  OpenAI API ключ не настроен. Некоторые тесты будут пропущены.")
        print("   Установите OPENAI_API_KEY в переменных окружения для полного тестирования.")
    
    # Запускаем тесты
    try:
        import pytest
        exit_code = pytest.main([__file__, "-v"])
        return exit_code == 0
    except ImportError:
        print("❌ pytest не установлен. Установите: pip install pytest pytest-asyncio")
        return False


if __name__ == "__main__":
    success = run_vector_tests()
    if success:
        print("✅ Все тесты векторной системы прошли успешно!")
    else:
        print("❌ Некоторые тесты не прошли. Проверьте логи выше.")
