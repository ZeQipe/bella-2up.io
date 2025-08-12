#!/usr/bin/env python3
"""
Скрипт для запуска тестов векторной системы
Можно запускать независимо от основного приложения
"""
import sys
import os

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_vector_system import run_vector_tests

if __name__ == "__main__":
    print("🚀 Запуск тестирования векторной системы...")
    print("=" * 60)
    
    success = run_vector_tests()
    
    print("=" * 60)
    if success:
        print("✅ Тестирование завершено успешно!")
        sys.exit(0)
    else:
        print("❌ Тестирование завершено с ошибками!")
        sys.exit(1)
