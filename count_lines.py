#!/usr/bin/env python3
"""
Подсчитываем сколько строк будет обрабатываться
"""
import os

def count_kb_lines():
    kb_path = "kb"
    total_lines = 0
    total_files = 0
    
    print("=== АНАЛИЗ ФАЙЛОВ KB ===")
    
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if not file.endswith('.txt'):
                continue
            if file.lower() == "promo.txt":  # Исключаем промо файл
                print(f"⏭️  Пропускаем: {file} (промо файл)")
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Разбиваем на строки и фильтруем как в коде
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                valid_lines = [line for line in lines if len(line) >= 10]
                
                print(f"📄 {file}:")
                print(f"   Всего строк: {len(lines)}")
                print(f"   Валидных строк (>=10 символов): {len(valid_lines)}")
                
                total_lines += len(valid_lines)
                total_files += 1
                
            except Exception as e:
                print(f"❌ Ошибка чтения {file}: {e}")
    
    print(f"\n📊 ИТОГО:")
    print(f"   Файлов: {total_files}")
    print(f"   Строк для обработки: {total_lines}")
    print(f"   Примерное время создания эмбеддингов: {total_lines * 0.1:.1f} секунд")
    
    return total_lines

if __name__ == "__main__":
    count_kb_lines()
