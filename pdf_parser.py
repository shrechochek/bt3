#!/usr/bin/env python3
"""
Улучшенный парсер PDF файлов с олимпиадными заданиями по биологии.
Извлекает задания с вариантами ответов, правильными ответами и изображениями.
"""

import fitz  # PyMuPDF
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def guess_ext_from_bytes(b: bytes) -> str:
    """Определяет расширение файла по сигнатуре байтов"""
    if b.startswith(b'\x89PNG\r\n\x1a\n'):
        return "png"
    if b.startswith(b'\xff\xd8\xff'):
        return "jpg"
    if b.startswith(b'GIF87a') or b.startswith(b'GIF89a'):
        return "gif"
    if b.startswith(b'II*\x00') or b.startswith(b'MM\x00*'):
        return "tiff"
    if b.startswith(b'BM'):
        return "bmp"
    if b[0:4] == b'RIFF' and b[8:12] == b'WEBP':
        return "webp"
    return "png"


def save_image_bytes(image_bytes: bytes, folder: str, base_name: str, ext: str = "jpg") -> str:
    """Сохраняет байты изображения в файл"""
    Path(folder).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(folder, f"{base_name}.{ext}")
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


class BiologyOlympiadParser:
    """Парсер PDF файлов с олимпиадными заданиями по биологии"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: str, images_output_dir: str = "images") -> List[Dict]:
        """
        Парсит PDF файл и возвращает список заданий

        Args:
            pdf_path: путь к PDF файлу
            images_output_dir: директория для сохранения изображений

        Returns:
            Список словарей с заданиями
        """
        self.logger.info(f"Начинаем парсинг PDF: {pdf_path}")

        # Открываем PDF
        doc = fitz.open(pdf_path)

        # Извлекаем текст всех страниц
        pages_text = [doc[p].get_text("text") or "" for p in range(len(doc))]
        full_text = "\n".join(pages_text)

        self.logger.info(f"Извлечено {len(pages_text)} страниц текста")

        # Извлекаем изображения
        images = self._extract_images(doc, images_output_dir)

        # Разбиваем текст на задания
        tasks = self._parse_tasks_from_text(full_text, images, images_output_dir, doc)

        doc.close()

        self.logger.info(f"Найдено {len(tasks)} заданий")
        return tasks

    def _extract_images(self, doc, images_output_dir: str) -> List[Dict]:
        """Извлекает все изображения из PDF"""
        images = []
        images_temp_dir = os.path.join(images_output_dir, "tmp")
        Path(images_temp_dir).mkdir(parents=True, exist_ok=True)

        for p_idx in range(len(doc)):
            page = doc[p_idx]
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") == 1:  # image block
                    bbox = tuple(block.get("bbox", [0, 0, 0, 0]))
                    img_obj = block.get("image")
                    image_bytes = None
                    ext = None

                    if isinstance(img_obj, dict):
                        xref = img_obj.get("xref")
                        if xref is not None:
                            try:
                                base = doc.extract_image(xref)
                                image_bytes = base.get("image")
                                ext = base.get("ext")
                            except:
                                image_bytes = None
                        else:
                            possible = img_obj.get("image")
                            if isinstance(possible, (bytes, bytearray)):
                                image_bytes = bytes(possible)
                    elif isinstance(img_obj, (bytes, bytearray)):
                        image_bytes = bytes(img_obj)
                    elif isinstance(img_obj, int):
                        try:
                            base = doc.extract_image(img_obj)
                            image_bytes = base.get("image")
                            ext = base.get("ext")
                        except:
                            image_bytes = None

                    if not image_bytes:
                        continue

                    if not ext:
                        ext = guess_ext_from_bytes(image_bytes)

                    temp_base = f"p{p_idx+1}_img{len(images)+1}"
                    temp_path = save_image_bytes(image_bytes, images_temp_dir, temp_base, ext)
                    images.append({
                        'temp_path': temp_path,
                        'page': p_idx,
                        'bbox': bbox,
                        'ext': ext
                    })

        self.logger.info(f"Извлечено {len(images)} изображений")
        return images

    def _parse_tasks_from_text(self, full_text: str, images: List[Dict], images_output_dir: str, doc) -> List[Dict]:
        """Парсит задания из полного текста PDF"""

        # Разбиваем текст на блоки заданий
        task_blocks = self._split_into_task_blocks(full_text)

        # Создаем анкоры для привязки изображений
        anchors = self._find_task_anchors(doc)

        tasks = []
        for block_num, block_text in task_blocks:
            task_data = self._parse_single_task(block_text, block_num, images, images_output_dir, anchors)
            if task_data:
                tasks.append(task_data)

        return tasks

    def _split_into_task_blocks(self, full_text: str) -> List[Tuple[int, str]]:
        """Разбивает полный текст на блоки отдельных заданий"""

        # Ищем все номера заданий (формат: "Задание 1", "Задание 1.1", etc.)
        task_pattern = r'(?:^|\n)\s*Задани(?:е|я)\s+([\d\.]{1,5})\s+'
        matches = list(re.finditer(task_pattern, full_text, re.IGNORECASE))

        task_blocks = []
        for i, match in enumerate(matches):
            task_num_str = match.group(1)
            # Для номеров типа 1.1 оставляем как строку
            task_num = task_num_str
            start_pos = match.start()

            # Определяем конец блока (начало следующего задания или конец текста)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(full_text)

            # Извлекаем текст блока
            block_text = full_text[start_pos:end_pos].strip()

            # Очищаем от "Задание N" в начале
            block_text = re.sub(r'^\s*Задани(?:е|я)\s+\d{1,3}\s+', '', block_text, count=1, flags=re.MULTILINE | re.IGNORECASE)

            task_blocks.append((task_num, block_text))

        self.logger.info(f"Найдено {len(task_blocks)} блоков заданий")
        return task_blocks

    def _find_task_anchors(self, doc) -> List[Dict]:
        """Находит анкоры (позиции заданий) в PDF для привязки изображений"""
        anchors = []

        for p_idx in range(len(doc)):
            # Ищем в текстовых блоках
            blocks = doc[p_idx].get_text("blocks")
            for b in blocks:
                textb = (b[4] or "").strip()
                for mm in re.finditer(r'Задани(?:е|я)\s+([\d\.]{1,5})', textb, re.IGNORECASE):
                    try:
                        num_str = mm.group(1)
                        # Оставляем как строку для точного сравнения
                        anchors.append({
                            'num': num_str,
                            'page': p_idx,
                            'x': (b[0] + b[2]) / 2.0,
                            'y': (b[1] + b[3]) / 2.0
                        })
                    except Exception:
                        continue

        # Убираем дубликаты анкоров
        uniq = []
        for a in anchors:
            if not any(u['num'] == a['num'] and u['page'] == a['page'] and abs(u['y'] - a['y']) < 3.0 for u in uniq):
                uniq.append(a)
        anchors = uniq

        self.logger.info(f"Найдено {len(anchors)} анкоров для привязки изображений")
        for anchor in anchors[:5]:  # Покажем первые 5 для отладки
            self.logger.debug(f"Анкор: {anchor}")
        return anchors

    def _parse_single_task(self, block_text: str, task_num, images: List[Dict], images_output_dir: str, anchors: List[Dict]) -> Optional[Dict]:
        """Парсит отдельное задание"""

        # Ищем правильный ответ в конце блока
        correct_answer = self._extract_correct_answer(block_text)

        # Удаляем упоминание правильного ответа из текста блока
        clean_block_text = block_text
        if correct_answer:
            # Удаляем строку с ответом
            clean_block_text = re.sub(r'(?i)(верный\s+)?ответ\s*[:\-]?.*$', '', clean_block_text, flags=re.MULTILINE).strip()

        if not correct_answer:
            # Если нет правильного ответа, возможно это задание открытого типа
            return self._parse_open_answer_task(clean_block_text, task_num, images, images_output_dir, anchors)

        # Разделяем на условие и варианты
        question_text, options = self._split_question_and_options(clean_block_text)

        if not options:
            # Если нет вариантов, возможно это тоже открытый тип
            return self._parse_open_answer_task(clean_block_text, task_num, images, images_output_dir, anchors)

        # Привязываем изображения к заданию
        task_images = self._assign_images_to_task(images, images_output_dir, task_num, anchors)

        return {
            'number': task_num,
            'question_text': question_text.strip(),
            'options': options,
            'correct_answer': correct_answer,
            'task_type': 'multiple_choice',
            'images': task_images
        }

    def _extract_correct_answer(self, block_text: str) -> Optional[str]:
        """Извлекает правильный ответ из блока текста"""

        # Ищем паттерны типа "Ответ: г", "Ответ: а)", "г)" и т.д.
        patterns = [
            r'(?i)ответ\s*[:\-]?\s*([а-яa-z])\s*\)?\.?',  # "Ответ: г"
            r'(?i)правильный\s+ответ\s*[:\-]?\s*([а-яa-z])\s*\)?\.?',  # "Правильный ответ: г"
            r'\b([а-яa-z])\s*\)\s*\.?\s*$',  # просто "г)" в конце строки
            r'\b([а-яa-z])\s*\.\s*$',  # просто "г." в конце строки
        ]

        for pattern in patterns:
            match = re.search(pattern, block_text, re.MULTILINE | re.IGNORECASE)
            if match:
                answer = match.group(1).lower().strip()
                # Проверяем что это буква варианта
                if answer in 'абвгдеёжзийклмнопрстуфхцчшщьыъэюяabcdefghijklmnopqrstuvwxyz':
                    return answer

        return None

    def _split_question_and_options(self, block_text: str) -> Tuple[str, List[str]]:
        """Разделяет блок на текст вопроса и варианты ответов"""

        lines = block_text.split('\n')

        # Ищем начало вариантов (строки типа "а)", "1.", "а.", etc.)
        option_start_idx = -1
        options = []
        current_option = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Проверяем, является ли строка началом варианта
            if self._is_option_start(line):
                # Сохраняем предыдущий вариант, если он был
                if current_option is not None:
                    options.append(current_option.strip())

                if option_start_idx == -1:
                    option_start_idx = i

                # Извлекаем текст варианта
                option_text = self._extract_option_text(line)
                if option_text:
                    current_option = option_text
                else:
                    current_option = line  # Если не удалось распознать, берем всю строку
            elif current_option is not None:
                # Продолжение текущего варианта
                current_option += ' ' + line

        # Сохраняем последний вариант
        if current_option is not None:
            options.append(current_option.strip())

        if option_start_idx == -1:
            # Нет вариантов, весь текст - вопрос
            return block_text, []

        # Разделяем на вопрос и варианты
        question_lines = lines[:option_start_idx]
        question_text = '\n'.join(question_lines).strip()

        return question_text, options

    def _is_option_start(self, line: str) -> bool:
        """Проверяет, является ли строка началом варианта ответа"""

        # Удаляем лишние пробелы
        s = line.strip()
        if not s:
            return False

        # Паттерны для вариантов:
        # а), б), в), г) - кириллица
        # a), b), c), d) - латиница
        # 1., 2., 3. - цифры
        # I., II., III. - римские цифры

        patterns = [
            r'^[а-яa-z]\s*\)\s+\S',  # а) текст, a) text
            r'^[а-яa-z]\.\s+\S',     # а. текст, a. text
            r'^\d{1,3}\s*\)\s+\S',   # 1) текст
            r'^\d{1,3}\.\s+\S',      # 1. текст
            r'^(?:[ivx]{1,4})\s*\)\s+\S',  # i), ii), iii)
            r'^(?:[ivx]{1,4})\.\s+\S',     # i., ii., iii.)
        ]

        for pattern in patterns:
            if re.match(pattern, s, re.IGNORECASE):
                return True

        return False

    def _extract_option_text(self, line: str) -> Optional[str]:
        """Извлекает текст варианта из строки"""

        # Удаляем маркер варианта
        patterns = [
            r'^[а-яa-z]\s*\)\s*(.*)',     # а) текст
            r'^[а-яa-z]\.\s*(.*)',        # а. текст
            r'^\d{1,3}\s*\)\s*(.*)',      # 1) текст
            r'^\d{1,3}\.\s*(.*)',         # 1. текст
            r'^(?:[ivx]{1,4})\s*\)\s*(.*)',  # i) текст
            r'^(?:[ivx]{1,4})\.\s*(.*)',     # i. текст
        ]

        for pattern in patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None  # Если не удалось распознать формат, возвращаем None

    def _parse_open_answer_task(self, block_text: str, task_num, images: List[Dict], images_output_dir: str, anchors: List[Dict]) -> Optional[Dict]:
        """Парсит задание открытого типа"""

        # Удаляем упоминание правильного ответа из текста вопроса
        question_text = re.sub(r'(?i)ответ\s*[:\-]?.*$', '', block_text, flags=re.MULTILINE).strip()

        # Ищем правильный ответ в конце
        correct_answer = self._extract_correct_answer(block_text)

        # Привязываем изображения
        task_images = self._assign_images_to_task(images, images_output_dir, task_num, anchors)

        return {
            'number': task_num,
            'question_text': question_text,
            'options': [],
            'correct_answer': correct_answer or '',
            'task_type': 'open_answer',
            'images': task_images
        }

    def _assign_images_to_task(self, images: List[Dict], images_output_dir: str, task_num, anchors: List[Dict]) -> List[str]:
        """Привязывает изображения к заданию на основе сегментов между анкорами"""

        # Находим анкор для этого задания
        task_anchor = None
        task_num_str = str(task_num)
        for anchor in anchors:
            if str(anchor['num']) == task_num_str:
                task_anchor = anchor
                break

        if not task_anchor:
            # Если не нашли анкор, возвращаем пустой список
            self.logger.debug(f"Не найден анкор для задания {task_num}")
            return []

        task_images = []
        task_page = task_anchor['page']
        task_y = task_anchor['y']

        # Находим следующий анкор на той же странице для определения границ сегмента
        next_anchor_y = float('inf')  # По умолчанию до конца страницы
        sorted_anchors_on_page = sorted([a for a in anchors if a['page'] == task_page], key=lambda a: a['y'])
        for i, anchor in enumerate(sorted_anchors_on_page):
            if str(anchor['num']) == task_num_str and i + 1 < len(sorted_anchors_on_page):
                next_anchor_y = sorted_anchors_on_page[i + 1]['y']
                break

        self.logger.debug(f"Задание {task_num}: сегмент y={task_y} до y={next_anchor_y}")

        # Фильтруем изображения, которые находятся в сегменте этого задания
        relevant_images = []
        for img in images:
            if img['page'] == task_page:
                # Изображение должно быть в сегменте задания (между анкором и следующим)
                img_mid_y = (img['bbox'][1] + img['bbox'][3]) / 2.0  # Середина изображения
                if task_y < img_mid_y < next_anchor_y:
                    relevant_images.append(img)
                    self.logger.debug(f"  Найдено изображение: mid_y={img_mid_y} в сегменте [{task_y}, {next_anchor_y}]")

        # Сортируем изображения по вертикальному положению
        relevant_images.sort(key=lambda img: (img['bbox'][1] + img['bbox'][3]) / 2.0)

        # Сохраняем изображения
        for idx, img in enumerate(relevant_images, start=1):
            safe_task_num = str(task_num).replace('.', '_')
            outbase = f"{safe_task_num}_{idx}"
            outpath = save_image_bytes(
                open(img['temp_path'], 'rb').read(),
                images_output_dir,
                outbase,
                img['ext']
            )
            task_images.append(os.path.basename(outpath))

        self.logger.debug(f"Заданию {task_num} присвоено {len(task_images)} изображений")
        return task_images


def test_parser():
    """Тестирование парсера на примерах"""

    parser = BiologyOlympiadParser()

    # Тестовый текст задания
    test_text = """
Активное использование антибиотиков в клинической практике и в сельском хозяйстве стало причиной возникновения серьёзной проблемы - антибиотикорезистентности. Это состояние, когда микроорганизмы теряют чувствительность к обычным антибиотикам. В результате появляются так называемые «суперпатогены», способные вызывать опасные инфекции, которые трудно лечить стандартными препаратами. Человечество оказалось втянутым в своеобразную «гонку вооружений» с микробами, где учёные постоянно ищут новые антибиотики, а микроорганизмы успешно адаптируются к ним. В настоящее время поиск новых антибиотиков осуществляют по нескольким направлениям, важнейшим из которых является почва. Ведь в ней обитает бесчисленное количество микроорганизмов, которые, к сожалению, плохо получается культивировать в лаборатории.

Но недавно большая команда учёных предложила решение проблемы культивирования таких бактерий, более того, они обнаружили очень интересный подштамм микроорганизма Eleftheria terrae ssp. carolina. Примечательно, что данная бактерия умеет выделять два вида антибиотиков, один из которых был ранее неизвестен. Оказалось, что он работает против антибиотикорезистентного золотистого стафилококка, Staphylococcus aureus, которым заражена примерно треть населения Земли. Антибиотик хорошо изучили и дали название кловибактин.

Как вы думаете, зачем бактерии может понадобиться два антибиотика и что будет, если отключить выработку одного?

а) Два антибиотика необходимы для того, чтобы эффективнее привлекать полезные микроорганизмы вступать в симбиотические отношения. Отключение выработки одного приведёт к снижению конкурентоспособности микроба.

б) Два антибиотика необходимы для более эффективной борьбы с плесневыми грибами. Отключение выработки одного антибиотика приведёт к ослаблению конкурентного потенциала бактерии.

в) Два антибиотика необходимы для оповещения других бактериях об изменении условий окружающей среды, температуры и влажности. Отключение выработки одного из них впоследствии может уничтожить часть популяции микробов.

г) Два антибиотика необходимы для более эффективной борьбы за ресурсы. Отключение выработки одного может привести к возникновению устойчивости к другому и последующему поражению микроба в конкурентной борьбе.

Ответ: г
"""

    task_blocks = parser._split_into_task_blocks("1. " + test_text)
    if task_blocks:
        task = parser._parse_single_task(task_blocks[0][1], task_blocks[0][0], [], "images")
        print("Parsed task:")
        print(f"Number: {task['number']}")
        print(f"Type: {task['task_type']}")
        print(f"Question: {task['question_text'][:200]}...")
        print(f"Options count: {len(task['options'])}")
        print(f"Correct answer: {task['correct_answer']}")


if __name__ == "__main__":
    test_parser()
