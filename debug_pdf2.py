#!/usr/bin/env python3
"""
Отладочный скрипт для анализа второго PDF файла
"""

import fitz
import re

def analyze_pdf(pdf_path):
    """Анализирует структуру PDF файла"""

    doc = fitz.open(pdf_path)
    print(f"PDF: {pdf_path}")
    print(f"Страниц: {len(doc)}")

    # Извлекаем текст всех страниц
    pages_text = [doc[p].get_text("text") or "" for p in range(len(doc))]
    full_text = "\n".join(pages_text)

    print(f"Общая длина текста: {len(full_text)} символов")

    # Ищем номера заданий
    task_patterns = [
        r'(?:^|\n)\s*(\d{1,3})\.\s+',
        r'Задани(?:е|я)\s*(\d{1,3})',
        r'№\s*(\d{1,3})',
    ]

    for pattern in task_patterns:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE))
        print(f"Паттерн '{pattern}': найдено {len(matches)} совпадений")
        if matches:
            for i, match in enumerate(matches[:3]):  # показываем первые 3
                start = max(0, match.start() - 30)
                end = min(len(full_text), match.end() + 30)
                context = full_text[start:end].replace('\n', ' ')
                print(f"  {i+1}: ...{context}...")

    # Ищем ответы
    answer_patterns = [
        r'Ответ\s*[:\-]?\s*[а-яa-z]',
        r'Правильный\s+ответ',
    ]

    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        print(f"Паттерн ответов '{pattern}': найдено {len(matches)} совпадений")

    # Показываем начало текста
    print("\nПервые 2000 символов текста:")
    print(repr(full_text[:2000]))

    # Ищем разделы
    section_patterns = [
        r'Часть\s+\d+',
        r'Максимальный\s+балл',
        r'Задание\s+\d+',
    ]

    for pattern in section_patterns:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        print(f"Разделы '{pattern}': найдено {len(matches)} совпадений")

    doc.close()

if __name__ == "__main__":
    analyze_pdf("tasks-biol-10-mun-msk-23-24.pdf")
