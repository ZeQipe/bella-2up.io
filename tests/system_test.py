#!/usr/bin/env python3
"""
Системный тест для проверки готовности бота к запуску
Проверяет все ключи API, векторную БД и основные компоненты
"""
import os
import sys
import asyncio
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config
from app.services.embedding_service import embedding_service, EmbeddingServiceError
from app.services.vector_service import vector_service, VectorServiceError
from app.services.ai_service import ai_service, AIServiceError
from app.services.prompt_service import prompt_service
from app.services.query_translator import query_translator
from app.database import db_manager


class SystemTester:
    """Системный тестер для проверки готовности бота"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.temp_dir = None
    
    def print_header(self, title: str):
        """Печатает заголовок секции"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print('='*60)
    
    def print_test(self, test_name: str, status: bool, details: str = ""):
        """Печатает результат теста"""
        emoji = "✅" if status else "❌"
        print(f"{emoji} {test_name}")
        if details:
            print(f"   {details}")
        self.results[test_name] = {"status": status, "details": details}
    
    def test_api_keys(self):
        """Тестирует наличие всех необходимых API ключей"""
        self.print_header("ПРОВЕРКА API КЛЮЧЕЙ")
        
        # OpenAI API Key
        openai_key = config.OPENAI_API_KEY
        self.print_test(
            "OpenAI API Key", 
            bool(openai_key and len(openai_key) > 10),
            f"Длина ключа: {len(openai_key) if openai_key else 0} символов"
        )
        
        # Telegram Bot Token
        tg_token = config.TELEGRAM_BOT_TOKEN
        self.print_test(
            "Telegram Bot Token",
            bool(tg_token and len(tg_token) > 20),
            f"Длина токена: {len(tg_token) if tg_token else 0} символов"
        )
        
        # DeepSeek API Key
        deepseek_key = config.DEEPSEEK_API_KEY
        self.print_test(
            "DeepSeek API Key",
            bool(deepseek_key and len(deepseek_key) > 10),
            f"Длина ключа: {len(deepseek_key) if deepseek_key else 0} символов"
        )
    
    async def test_embedding_service(self):
        """Тестирует сервис эмбеддингов"""
        self.print_header("ПРОВЕРКА СЕРВИСА ЭМБЕДДИНГОВ")
        
        if not config.OPENAI_API_KEY:
            self.print_test("OpenAI Embeddings", False, "API ключ не настроен")
            return
        
        try:
            # Тест создания эмбеддинга
            test_text = "Как зарегистрироваться в казино?"
            embedding = await embedding_service.create_embedding(test_text)
            
            self.print_test(
                "Создание эмбеддинга",
                isinstance(embedding, list) and len(embedding) > 0,
                f"Размерность: {len(embedding)} чисел"
            )
            
            # Тест health check
            health = await embedding_service.health_check()
            self.print_test("Health Check OpenAI", health, "API доступен" if health else "API недоступен")
            
        except EmbeddingServiceError as e:
            self.print_test("OpenAI Embeddings", False, f"Ошибка API: {str(e)}")
        except Exception as e:
            self.print_test("OpenAI Embeddings", False, f"Неожиданная ошибка: {str(e)}")
    
    async def test_vector_database(self):
        """Тестирует векторную базу данных"""
        self.print_header("ПРОВЕРКА ВЕКТОРНОЙ БАЗЫ ДАННЫХ")
        
        if not config.OPENAI_API_KEY:
            self.print_test("Vector Database", False, "OpenAI API ключ не настроен")
            return
        
        # Создаем временную директорию для теста
        self.temp_dir = tempfile.mkdtemp()
        temp_chroma_dir = os.path.join(self.temp_dir, "test_chroma")
        temp_kb_dir = os.path.join(self.temp_dir, "test_kb")
        
        try:
            # Создаем тестовую базу знаний на английском языке
            os.makedirs(temp_kb_dir, exist_ok=True)
            test_kb_file = os.path.join(temp_kb_dir, "test_en.txt")
            with open(test_kb_file, 'w', encoding='utf-8') as f:
                f.write("Casino registration takes 5 minutes to complete\n")
                f.write("Minimum deposit amount is 500 rubles\n")
                f.write("Withdrawal processing takes 24 hours\n")
                f.write("Account verification is required for first withdrawal\n")
                f.write("Welcome bonus is 100% of first deposit\n")
            
            # Настраиваем временные пути
            original_db_path = vector_service.db_path
            vector_service.db_path = temp_chroma_dir
            
            # Тест инициализации базы знаний
            await vector_service.initialize_knowledge_base(temp_kb_dir)
            
            document_count = vector_service.collection.count() if vector_service.collection else 0
            self.print_test(
                "Инициализация векторной БД",
                document_count > 0,
                f"Создано {document_count} векторов"
            )
            
            # Тест поиска с переводом запроса
            if document_count > 0:
                # Тестируем русский запрос (должен переводиться на английский)
                search_results = await vector_service.search_similar("registration process", limit=3)
                self.print_test(
                    "Поиск в векторной БД (английский)",
                    len(search_results) > 0,
                    f"Найдено {len(search_results)} релевантных документов"
                )
                
                # Показываем результаты
                if search_results:
                    first_result = search_results[0]
                    similarity = first_result.get('similarity', 0)
                    content = first_result.get('content', '')[:60]
                    print(f"   Лучший результат: {similarity:.3f} - '{content}...'")
            
            # Тест добавления документа
            doc_id = await vector_service.add_document(
                "Тестовый документ для проверки", 
                {"test": True}
            )
            self.print_test(
                "Добавление документа",
                bool(doc_id),
                f"ID документа: {doc_id}"
            )
            
            # Принудительно закрываем ChromaDB соединение
            if vector_service.client:
                try:
                    vector_service.client = None
                    vector_service.collection = None
                    vector_service._initialized = False
                except Exception:
                    pass
            
            # Восстанавливаем оригинальный путь
            vector_service.db_path = original_db_path
            
        except VectorServiceError as e:
            self.print_test("Vector Database", False, f"Ошибка векторной БД: {str(e)}")
        except Exception as e:
            self.print_test("Vector Database", False, f"Неожиданная ошибка: {str(e)}")
    
    def test_database(self):
        """Тестирует SQLite базу данных"""
        self.print_header("ПРОВЕРКА БАЗЫ ДАННЫХ SQLite")
        
        try:
            # Тест подключения к БД
            with db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
            
            self.print_test(
                "Подключение к SQLite",
                table_count >= 2,  # Ожидаем минимум 2 таблицы
                f"Найдено {table_count} таблиц"
            )
            
            # Тест создания тестового чата
            from app.models import ChatState
            from app.config import PersonaType
            
            test_chat_id = 999999999  # Тестовый ID
            test_state = ChatState.create_default(test_chat_id)
            db_manager.save_chat_state(test_state)
            
            # Проверяем что состояние сохранилось
            retrieved_state = db_manager.get_chat_state(test_chat_id)
            self.print_test(
                "Сохранение состояния чата",
                retrieved_state.chat_id == test_chat_id,
                f"Персона: {retrieved_state.current_persona.value}"
            )
            
        except Exception as e:
            self.print_test("SQLite Database", False, f"Ошибка БД: {str(e)}")
    
    def test_prompt_service(self):
        """Тестирует сервис промтов"""
        self.print_header("ПРОВЕРКА СЕРВИСА ПРОМТОВ")
        
        try:
            from app.config import PersonaType
            
            # Тест загрузки промтов для всех персон
            personas = [PersonaType.BUSINESS, PersonaType.BELLA, PersonaType.BEN]
            
            for persona in personas:
                prompt = prompt_service.get_system_prompt(
                    persona=persona,
                    promotions_info="Тестовые промо",
                    context_info="Тестовый контекст"
                )
                
                self.print_test(
                    f"Промт для {persona.value}",
                    len(prompt) > 50,
                    f"Длина промта: {len(prompt)} символов"
                )
            
        except Exception as e:
            self.print_test("Prompt Service", False, f"Ошибка сервиса промтов: {str(e)}")
    
    async def test_query_translator(self):
        """Тестирует сервис перевода запросов"""
        self.print_header("ПРОВЕРКА СЕРВИСА ПЕРЕВОДА ЗАПРОСОВ")
        
        try:
            # Тест перевода обычного запроса
            translated = await query_translator.translate_query("как зарегистрироваться в казино?")
            self.print_test(
                "Перевод казино-запроса",
                translated is not None and len(translated) > 0,
                f"'{translated}'" if translated else "Не переведено"
            )
            
            # Тест определения casual chat
            casual_result = await query_translator.translate_query("привет красавчик, как дела?")
            self.print_test(
                "Определение casual chat",
                casual_result is None,
                "Корректно определен как обычная болтовня" if casual_result is None else f"Неожиданный результат: {casual_result}"
            )
            
            # Тест health check
            health = await query_translator.health_check()
            self.print_test("Health Check переводчика", health, "Сервис доступен")
            
        except Exception as e:
            self.print_test("Query Translator", False, f"Ошибка переводчика: {str(e)}")
    
    async def test_ai_service(self):
        """Тестирует AI сервис (заглушку)"""
        self.print_header("ПРОВЕРКА AI СЕРВИСА")
        
        try:
            from app.config import PersonaType
            
            # Тест генерации ответа с векторным контекстом
            test_chat_id = 999999999
            print("\n🤖 Тестируем AI с векторным контекстом...")
            
            response = await ai_service.generate_response(
                chat_id=test_chat_id,
                current_message="как зарегистрироваться?",
                persona=PersonaType.BUSINESS
            )
            
            self.print_test(
                "Генерация ответа с контекстом",
                len(response) > 0,
                f"Ответ: '{response[:100]}...'"
            )
            
            # Тест casual chat
            print("\n💬 Тестируем casual chat...")
            casual_response = await ai_service.generate_response(
                chat_id=test_chat_id,
                current_message="привет красавчик!",
                persona=PersonaType.BELLA
            )
            
            self.print_test(
                "Генерация ответа (casual chat)",
                len(casual_response) > 0,
                f"Ответ: '{casual_response[:100]}...'"
            )
            
            # Тест health check
            health = await ai_service.health_check()
            self.print_test("Health Check AI", health, "Сервис доступен")
            
        except AIServiceError as e:
            self.print_test("AI Service", False, f"Ошибка AI: {str(e)}")
        except Exception as e:
            self.print_test("AI Service", False, f"Неожиданная ошибка: {str(e)}")
    
    def cleanup(self):
        """Очистка временных файлов"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Даем время ChromaDB закрыть файлы
                import time
                time.sleep(1)
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"⚠️  Не удалось очистить временные файлы: {e}")
                print("   Это не влияет на работу бота, только на очистку тестовых данных.")
    
    def print_summary(self):
        """Печатает итоговый отчет"""
        self.print_header("ИТОГОВЫЙ ОТЧЕТ")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["status"])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Всего тестов: {total_tests}")
        print(f"✅ Успешно: {passed_tests}")
        print(f"❌ Неудачно: {failed_tests}")
        print(f"📈 Процент успеха: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n⚠️  НЕУДАЧНЫЕ ТЕСТЫ:")
            for test_name, result in self.results.items():
                if not result["status"]:
                    print(f"   • {test_name}: {result['details']}")
        
        print(f"\n{'='*60}")
        
        if failed_tests == 0:
            print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Бот готов к запуску!")
            return True
        else:
            print("🚨 ЕСТЬ ПРОБЛЕМЫ! Исправьте ошибки перед запуском бота.")
            return False


async def main():
    """Основная функция тестирования"""
    print("🚀 СИСТЕМНЫЙ ТЕСТ ГОТОВНОСТИ БОТА")
    print(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = SystemTester()
    
    try:
        # Запускаем все тесты
        tester.test_api_keys()
        await tester.test_embedding_service()
        await tester.test_vector_database()
        tester.test_database()
        tester.test_prompt_service()
        await tester.test_query_translator()
        await tester.test_ai_service()
        
        # Показываем итоги
        success = tester.print_summary()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Тестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"💥 ФАТАЛЬНАЯ ОШИБКА: {str(e)}")
        sys.exit(1)
