#!/usr/bin/env python3
"""
Скрипт для импорта PDF файлов с олимпиадными заданиями в базу данных.
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path
from pdf_parser import BiologyOlympiadParser

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из app.py
from app import save_task_options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFImporter:
    """Импортер PDF файлов в базу данных"""

    def __init__(self, db_path: str = "datbase.db", images_dir: str = "images"):
        self.db_path = db_path
        self.images_dir = images_dir
        self.parser = BiologyOlympiadParser()

        # Создаем директорию для изображений
        Path(images_dir).mkdir(parents=True, exist_ok=True)

    def import_pdf(self, pdf_path: str) -> int:
        """
        Импортирует PDF файл в базу данных

        Args:
            pdf_path: путь к PDF файлу

        Returns:
            количество импортированных заданий
        """
        logger.info(f"Начинаем импорт PDF: {pdf_path}")

        # Парсим PDF
        tasks = self.parser.parse_pdf(pdf_path, self.images_dir)

        if not tasks:
            logger.warning("Не найдено ни одного задания в PDF")
            return 0

        # Сохраняем задания в базу данных
        saved_count = self._save_tasks_to_db(tasks, pdf_path)

        logger.info(f"Импортировано {saved_count} заданий из {len(tasks)} найденных")
        return saved_count

    def _save_tasks_to_db(self, tasks: list, source_pdf: str) -> int:
        """Сохраняет задания в базу данных"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        saved_count = 0

        try:
            for task in tasks:
                try:
                    # Формируем данные для сохранения
                    number = task.get('number')
                    title = f"Задание {number}" if number is not None else "Задание"
                    description = task['question_text']
                    task_type = task['task_type']

                    # Для multiple_choice правильный ответ хранится в опциях
                    if task_type == 'multiple_choice':
                        answer_text = ""
                    else:
                        answer_text = task.get('correct_answer', '')

                    difficulty = 1
                    tags = ""
                    source = os.path.basename(source_pdf)

                    is_visible_to_students = 0

                    # Сохраняем задание в БД
                    cursor.execute(
                        """INSERT INTO tasks6
                           (title, description, anwser, difficulty, tags, source, task_type, is_visible_to_students)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (title, description, answer_text, difficulty, tags, source, task_type, is_visible_to_students)
                    )

                    task_id = cursor.lastrowid

                    # Сохраняем изображения (переименовываем)
                    for idx, imgfile in enumerate(task.get('images', []), start=1):
                        src = os.path.join(self.images_dir, imgfile)
                        if os.path.exists(src):
                            ext = os.path.splitext(src)[1] or ".jpg"
                            dest = os.path.join(self.images_dir, f"{task_id}_{idx}{ext}")
                            os.replace(src, dest)

                    # Для multiple_choice сохраняем варианты
                    if task_type == 'multiple_choice':
                        options = task.get('options', [])
                        correct_raw = task.get('correct_answer', "")

                        # Определяем правильный вариант
                        correct_options = []
                        if correct_raw:
                            # Ищем вариант, который соответствует правильному ответу
                            for i, option_text in enumerate(options):
                                # Определяем букву варианта (а, б, в, г, etc.)
                                option_letter = chr(ord('а') + i)  # а=0, б=1, в=2, г=3, etc.
                                if option_letter == correct_raw.lower():
                                    correct_options.append(option_text)
                                    break

                        # Сохраняем опции
                        options_data = []
                        for opt_text in options:
                            is_corr = opt_text in correct_options
                            options_data.append({'text': opt_text, 'is_correct': is_corr})

                        if options_data:
                            save_task_options(task_id, options_data, cursor=cursor)

                    saved_count += 1
                    logger.debug(f"Сохранено задание {task_id} (номер {number})")

                except Exception as e:
                    logger.error(f"Ошибка сохранения задания {task.get('number')}: {e}")
                    logger.error(f"Task data: {task}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            conn.commit()
            logger.info(f"Успешно сохранено {saved_count} заданий")

        except Exception as e:
            logger.error(f"Ошибка при сохранении в базу данных: {e}")
            conn.rollback()
        finally:
            conn.close()

        return saved_count


def main():
    """Основная функция для запуска импорта"""

    if len(sys.argv) < 2:
        print("Использование: python pdf_importer.py <путь_к_pdf>")
        print("Пример: python pdf_importer.py ans-biol-9-sch-msk-23-24.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Файл не найден: {pdf_path}")
        sys.exit(1)

    # Импортируем
    importer = PDFImporter()
    try:
        count = importer.import_pdf(pdf_path)
        print(f"Импорт завершен. Импортировано {count} заданий.")
    except Exception as e:
        logger.error(f"Ошибка импорта: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
