import glob
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, make_response, abort, send_from_directory, current_app
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
import os
from datetime import datetime, timedelta
import hashlib
import re
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import json

app = Flask(__name__)
app.secret_key = 'SECRET_KEY'
app.config['UPLOAD_FOLDER'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {"pdf"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
DB_PATH = "datbase.db"

def parse_int_or_none(value):
    try:
        return int(value) if value and str(value).strip() else None
    except Exception:
        return None
    
def parse_pdf_and_store_tasks(pdf_path, db_path, images_output_dir="images"):
    # Открываем PDF и читаем текст всех страниц
    doc = fitz.open(pdf_path)
    pages_text = [page.get_text("text") or "" for page in doc]
    full_text = "\n".join(pages_text)
    # Собираем якори номеров задач (номер, страница, координаты)
    anchors = []
    for p_idx, page in enumerate(doc):
        words = page.get_text("words")
        for w in words:
            token = w[4].strip()
            m = re.match(r'^(?:№\s*)?(\d{1,3})[.)]$', token)
            if m:
                anchors.append({'num': int(m.group(1)), 'page': p_idx,
                                'x': (w[0]+w[2])/2, 'y': (w[1]+w[3])/2})
        blocks = page.get_text("blocks")
        for b in blocks:
            textb = (b[4] or "").strip()
            for mm in re.finditer(r'Задани(?:е|я)\s*(\d{1,3})', textb, re.IGNORECASE):
                anchors.append({'num': int(mm.group(1)), 'page': p_idx,
                                'x': (b[0]+b[2])/2, 'y': (b[1]+b[3])/2})
    # Убираем дубликаты анкоров с близкими координатами
    uniq = []
    for a in anchors:
        if not any(a['num']==u['num'] and a['page']==u['page'] and abs(a['y']-u['y'])<3.0 
                   for u in uniq):
            uniq.append(a)
    anchors = uniq

    # Извлекаем все изображения во временную папку
    temp_img_dir = os.path.join(images_output_dir, "tmp")
    Path(temp_img_dir).mkdir(parents=True, exist_ok=True)
    images = []  # список словарей {'page', 'bbox', 'temp_path', 'ext'}
    for p_idx, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:  # блок типа "картинка"
                bbox = tuple(block.get("bbox", [0,0,0,0]))
                img_obj = block.get("image")
                image_bytes = None
                ext = None
                # Извлекаем байты изображения из блока
                if isinstance(img_obj, dict) and img_obj.get("xref") is not None:
                    try:
                        base = doc.extract_image(img_obj["xref"])
                        image_bytes = base["image"]
                        ext = base["ext"]
                    except:
                        pass
                elif isinstance(img_obj, (bytes, bytearray)):
                    image_bytes = bytes(img_obj)
                if not image_bytes:
                    continue
                if not ext:
                    # если расширение не задано, определяем по сигнатуре
                    if image_bytes.startswith(b'\x89PNG'):
                        ext = "png"
                    elif image_bytes.startswith(b'\xff\xd8\xff'):
                        ext = "jpg"
                    else:
                        ext = "jpg"
                # Сохраняем временно
                tmp_name = f"p{p_idx+1}_img{len(images)+1}"
                temp_path = os.path.join(temp_img_dir, tmp_name + "." + ext)
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)
                images.append({'page': p_idx, 'bbox': bbox,
                               'temp_path': temp_path, 'ext': ext})

    # Разбиваем полный текст на блоки заданий по номерам
    starts = [(m.start(), int(m.group(1))) 
              for m in re.finditer(r'(?m)^\s*(\d{1,3})\.\s*', full_text)]
    if not starts:
        # если не нашли по всему тексту, попробуем постранично
        cursor = 0
        for txt in pages_text:
            for m in re.finditer(r'(?m)^\s*(\d{1,3})\.\s*', txt):
                starts.append((cursor + m.start(), int(m.group(1))))
            cursor += len(txt) + 1
    starts.sort()
    task_blocks = []
    for i, (pos, num) in enumerate(starts):
        start = pos
        end = starts[i+1][0] if i+1 < len(starts) else len(full_text)
        block = full_text[start:end].strip()
        block = re.sub(r'^\s*\d{1,3}\.\s*', '', block, count=1)
        task_blocks.append({'num': num, 'text': block})

    # Парсим каждый блок: вопрос и варианты
    parsed = []
    for tb in task_blocks:
        txt = tb['text']
        m_ans = re.search(r'(?mi)\bОтвет\b[:\s]*', txt)
        if m_ans:
            question = txt[:m_ans.start()].strip()
            answers_text = txt[m_ans.end():].strip()
        else:
            # нет «Ответ:» — разделим по пустой строке или весь блок — вопрос без вариантов
            parts = re.split(r'\n\s*\n', txt, maxsplit=1)
            question = parts[0].strip()
            answers_text = parts[1].strip() if len(parts)>1 else ""
        # Очищаем вопрос от лишних постраничных номеров
        question = re.sub(r'\n+\d{1,3}\s*$', '', question).strip()
        # Разбираем варианты
        answers = []
        correct = []
        if answers_text:
            for line in answers_text.splitlines():
                raw = line.strip()
                if not raw or re.match(r'^\d{1,3}$', raw):
                    continue
                # Удаляем маркеры (пример функции clean_option_line)
                is_correct = bool(re.search(r'[✓✔]', raw))
                line_clean = re.sub(r'[✓✔]', '', raw)
                line_clean = re.sub(r'^[\s\-–—\*\u2022\u25CF]+', '', line_clean)
                line_clean = re.sub(r'^[A-Za-zА-Яа-я]\s*[\.\)]\s*', '', line_clean)
                line_clean = re.sub(r'^\d{1,3}\s*[\.\)]\s*', '', line_clean).strip()
                if line_clean:
                    answers.append(line_clean)
                    if is_correct:
                        correct.append(line_clean)
        parsed.append({'num': tb['num'], 'question': question,
                       'answers': answers, 'correct': correct, 'images': []})

    # Привязка картинок к заданиям по координатам
    def img_mid_y(bbox): return (bbox[1] + bbox[3]) / 2.0
    num_to_idx = {p['num']: i for i,p in enumerate(parsed) if p['num'] is not None}
    for img in images:
        img_page = img['page']
        mid_y = img_mid_y(img['bbox'])
        # ищем ближайший анкор
        best_anchor = None
        best_score = None
        for anc in anchors:
            score = abs(img_page - anc['page'])*10000 + abs(mid_y - anc['y'])
            # Если картинка значительно выше анкера, добавляем штраф
            if mid_y + 20 < anc['y']:
                score += 2000
            if best_score is None or score < best_score:
                best_score = score
                best_anchor = anc
        assigned = best_anchor['num'] if best_anchor else None
        if assigned in num_to_idx:
            idx = num_to_idx[assigned]
        else:
            idx = 0  # fallback к первому заданию
        task = parsed[idx]
        # Сохраняем картинку в выходную папку images_output_dir
        existing = task['images']
        out_base = f"{task['num']}_{len(existing)+1}"
        dest_path = save_image_bytes(open(img['temp_path'],'rb').read(),
                                     images_output_dir, out_base, img['ext'])
        task['images'].append(os.path.basename(dest_path))

    # Очищаем временную папку картинок
    for f in glob.glob(os.path.join(temp_img_dir, "*")):
        try: os.remove(f)
        except: pass
    try: os.rmdir(temp_img_dir)
    except: pass

    # Сохраняем задания в БД
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    created = []
    for t in parsed:
        title = f"Задание {t['num']}" if t.get('num') else "Задание"
        description = t['question']
        # Сохраняем список ответов/правильный можно по желанию
        # Например, сохраняем правильный ответ в колонку 'anwser'
        answer_text = t['correct'][0] if t['correct'] else ""
        cur.execute("INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source) VALUES (?, ?, ?, ?, ?, ?)",
                    (title, description, answer_text, 1, "", os.path.basename(pdf_path)))
        task_id = cur.lastrowid
        # Перемещаем файлы изображений в папку images/, присваивая им имена task_id_N.jpg
        for idx, imgfile in enumerate(t['images'], start=1):
            src = os.path.join(images_output_dir, imgfile)
            if os.path.exists(src):
                ext = os.path.splitext(src)[1] or ".jpg"
                dest = os.path.join(images_output_dir, f"{task_id}_{idx}{ext}")
                os.replace(src, dest)
        created.append({"task_id": task_id, "question": description,
                        "answers": t['answers'], "correct": t['correct']})
    conn.commit()
    conn.close()
    return created

def search_tasks(params, page=1, per_page=20):
    where_clauses = []
    args = []
    if params.get('title'):
        where_clauses.append("title LIKE ?")
        args.append(f"%{params['title']}%")
    if params.get('description'):
        where_clauses.append("description LIKE ?")
        args.append(f"%{params['description']}%")
    if params.get('tag'):
        where_clauses.append("tags LIKE ?")
        args.append(f"%{params['tag']}%")
    if params.get('source'):
        where_clauses.append("source LIKE ?")
        args.append(f"%{params['source']}%")
    if params.get('task_id'):
        task_id = parse_int_or_none(params.get('task_id'))
        if task_id is not None:
            where_clauses.append("rowid = ?")
            args.append(task_id)
    min_task_id = parse_int_or_none(params.get('min_task_id'))
    max_task_id = parse_int_or_none(params.get('max_task_id'))
    if min_task_id is not None:
        where_clauses.append("rowid >= ?")
        args.append(min_task_id)
    if max_task_id is not None:
        where_clauses.append("rowid <= ?")
        args.append(max_task_id)
    min_d = parse_int_or_none(params.get('min_difficulty'))
    max_d = parse_int_or_none(params.get('max_difficulty'))
    if min_d is not None:
        where_clauses.append("difficulty >= ?")
        args.append(min_d)
    if max_d is not None:
        where_clauses.append("difficulty <= ?")
        args.append(max_d)
    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    
    # Calculate offset for pagination
    offset = (page - 1) * per_page
    
    # First, get total count for pagination info
    count_sql = f"SELECT COUNT(*) FROM tasks6 {where_sql}"
    
    # Then get the paginated results
    sql = f"SELECT rowid, title, description, anwser, difficulty, tags, source FROM tasks6 {where_sql} ORDER BY rowid ASC LIMIT ? OFFSET ?"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Get total count
        cur.execute(count_sql, args)
        total_count = cur.fetchone()[0]
        
        # Get paginated results
        cur.execute(sql, args + [per_page, offset])
        results = cur.fetchall()
        
        return {
            'tasks': results,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page  # Ceiling division
        }
    except Exception as e:
        current_app.logger.exception("Ошибка при поиске задач")
        return {
            'tasks': [],
            'total_count': 0,
            'page': 1,
            'per_page': per_page,
            'total_pages': 0
        }
    finally:
        conn.close()

def fetch_user(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, username, name, surname, password FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return {'id': row[0], 'username': row[1], 'name': row[2], 'surname': row[3], 'password': row[4]} if row else None
    except Exception:
        current_app.logger.exception("Ошибка получения пользователя")
        return None

def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def get_difficulty_color(difficulty):
    colors = ['#3498db', '#2ecc71', '#1abc9c', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#2c3e50']
    return colors[max(0, min(int(difficulty) - 1, len(colors) - 1))]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_pdf_to_tasks_clean(pdf_path, images_output_dir="/mnt/data/parsed_images_clean", save_images=True):
    """
    Разбивает PDF на задания:
      - находит номера заданий (1., 2) и т.п.
      - отделяет текст вопроса до слова 'Ответ'
      - извлекает варианты ответов и помечает отмеченные галочкой
      - извлекает изображения и привязывает их к ближайшему заданию по координатам
    Возвращает список dict'ов для каждого задания.
    Сохраняет изображения в images_output_dir (имена: <номер_задания>_1.ext, ...).
    """
    doc = fitz.open(pdf_path)
    pages_text = [doc[p].get_text("text") or "" for p in range(len(doc))]
    full_text = "\n".join(pages_text)

    # 1) anchors: позиции, где в тексте встречается номер задания (по словам на страницах)
    anchors = []
    for p_idx in range(len(doc)):
        words = doc[p_idx].get_text("words")  # x0,y0,x1,y1,word,...
        for w in words:
            token = w[4].strip()
            m = re.match(r'^(?:№\s*)?(\d{1,3})([.)\uFF09]?)$', token)
            if m:
                anchors.append({'num': int(m.group(1)), 'page': p_idx, 'x': (w[0]+w[2])/2.0, 'y': (w[1]+w[3])/2.0})
        # также попытки поймать строки вроде "Задание 12"
        blocks = doc[p_idx].get_text("blocks")
        for b in blocks:
            textb = (b[4] or "").strip()
            for mm in re.finditer(r'Задани(?:е|я)\s*(\d{1,3})', textb, re.IGNORECASE):
                anchors.append({'num': int(mm.group(1)), 'page': p_idx, 'x': (b[0]+b[2])/2.0, 'y': (b[1]+b[3])/2.0})

    # deduplicate anchors (по num,page,y близко)
    uniq = []
    for a in anchors:
        dup = False
        for u in uniq:
            if u['num']==a['num'] and u['page']==a['page'] and abs(u['y']-a['y'])<3.0:
                dup = True; break
        if not dup:
            uniq.append(a)
    anchors = uniq

    # 2) Извлечь изображения (временно в подпапку), собрать bbox и страницу
    images_temp_dir = os.path.join(images_output_dir, "tmp")
    Path(images_temp_dir).mkdir(parents=True, exist_ok=True)
    images = []
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:
                bbox = tuple(block.get("bbox", [0,0,0,0]))
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
                        except Exception:
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
                    except Exception:
                        image_bytes = None
                if not image_bytes:
                    continue
                if not ext:
                    ext = guess_ext_from_bytes(image_bytes)
                temp_base = f"p{p_idx+1}_img{len(images)+1}"
                temp_path = save_image_bytes(image_bytes, images_temp_dir, temp_base, ext)
                images.append({'temp_path': temp_path, 'page': p_idx, 'bbox': bbox, 'ext': ext})

    # 3) Разбивка текста на блоки заданий по номерам "N." или "N)"
    starts = [(m.start(), int(m.group(1))) for m in re.finditer(r'(?m)^\s*(\d{1,3})[.)]\s*', full_text)]
    tasks_blocks = []
    if not starts:
        # fallback постранично
        cursor = 0
        for p_idx, txt in enumerate(pages_text):
            for m in re.finditer(r'(?m)^\s*(\d{1,3})[.)]\s*', txt):
                starts.append((cursor + m.start(), int(m.group(1))))
            cursor += len(txt) + 1
    if starts:
        for i,(pos,num) in enumerate(starts):
            start_pos = pos
            end_pos = starts[i+1][0] if i+1 < len(starts) else len(full_text)
            block = full_text[start_pos:end_pos].strip()
            # убираем ведущую "N." в начале блока
            block = re.sub(r'^\s*\d{1,3}[.)]\s*', '', block, count=1)
            back = full_text[max(0, start_pos-300):start_pos]
            mbl = re.search(r'Блок\s*№\s*(\d+)', back)
            block_label = f"Блок №{mbl.group(1)}" if mbl else None
            tasks_blocks.append({'num': num, 'start': start_pos, 'end': end_pos, 'text': block, 'block_label': block_label})
    else:
        tasks_blocks.append({'num': None, 'start':0, 'end':len(full_text), 'text': full_text, 'block_label': None})

    # 4) Для каждого блока выделяем вопрос (до 'Ответ') и варианты (после)
    parsed = []
    for tb in tasks_blocks:
        txt = tb['text']
        m_ans = re.search(r'(?mi)\bОтвет\b[:\s]*', txt)
        if m_ans:
            q_text = txt[:m_ans.start()].strip()
            a_text = txt[m_ans.end():].strip()
        else:
            parts = re.split(r'\n{2,}', txt, maxsplit=1)
            if len(parts) > 1:
                q_text, a_text = parts[0].strip(), parts[1].strip()
            else:
                q_text, a_text = txt.strip(), ""
        # очистка вопроса от лишних цифр в конце (страницы) и пробелов
        q_text = re.sub(r'\n+\d{1,3}\s*$', '', q_text).strip()
        # варианты: по строкам, чистим маркеры
        answers = []
        correct = []
        if a_text:
            for line in a_text.splitlines():
                raw = line.strip()
                if not raw: 
                    continue
                if re.match(r'^\d{1,3}$', raw):  # вероятно номер страницы
                    continue
                cleaned, is_correct = clean_option_line(raw)
                if cleaned:
                    answers.append(cleaned)
                    if is_correct:
                        correct.append(cleaned)
        parsed.append({
            'block': tb.get('block_label'),
            'number': tb.get('num'),
            'question': q_text,
            'answers_text': a_text,
            'answers': answers,
            'correct': correct,
            'images': []
        })

    # 5) Привязка изображений к заданиям: ищем ближайший anchor и назначаем изображение соответствующему номеру задания
    num_to_index = {p['number']: i for i,p in enumerate(parsed) if p['number'] is not None}
    def img_mid_y(bbox): return (bbox[1] + bbox[3]) / 2.0
    for img in images:
        best = None; best_score = None
        for a in anchors:
            s = abs(img['page'] - a['page']) * 10000 + abs(img_mid_y(img['bbox']) - a['y'])
            if img_mid_y(img['bbox']) + 20 < a['y']:
                s += 2000
            if best_score is None or s < best_score:
                best_score = s; best = a
        assigned_num = best['num'] if best else None
        if assigned_num in num_to_index:
            idx = num_to_index[assigned_num]
        else:
            # fallback: просто прикрепим к первому (если не нашли)
            idx = 0
        tasknum = parsed[idx]['number'] if parsed[idx]['number'] is not None else f"i{idx+1}"
        existing = parsed[idx]['images']
        outbase = f"{tasknum}_{len(existing)+1}"
        outpath = save_image_bytes(open(img['temp_path'],'rb').read(), images_output_dir, outbase, img['ext'])
        parsed[idx]['images'].append(os.path.basename(outpath))

    # Cleanup временной папки
    try:
        for f in glob.glob(os.path.join(images_temp_dir, "*")):
            try: os.remove(f)
            except: pass
        try: os.rmdir(images_temp_dir)
        except: pass
    except:
        pass

    return parsed

def guess_ext_from_bytes(b: bytes) -> str:
    if b.startswith(b'\x89PNG\r\n\x1a\n'): return "png"
    if b.startswith(b'\xff\xd8\xff'): return "jpg"
    if b.startswith(b'GIF87a') or b.startswith(b'GIF89a'): return "gif"
    if b.startswith(b'II*\x00') or b.startswith(b'MM\x00*'): return "tiff"
    if b.startswith(b'BM'): return "bmp"
    if b[0:4] == b'RIFF' and b[8:12] == b'WEBP': return "webp"
    return "png"

def save_image_bytes(image_bytes: bytes, folder: str, base_name: str, ext: str = "jpg"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(folder, f"{base_name}.{ext}")
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path

def get_test_by_id(test_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tests WHERE id = ?", (test_id,))
    test = cursor.fetchone()
    conn.close()
    return test

def get_task_images(task_id):
    """
    Возвращает список имён файлов (без пути) в app.config['UPLOAD_FOLDER'],
    относящихся к задаче task_id.
    Поддерживаются имена:
      - "<task_id>_1.jpg", "<task_id>_2.png", ...
      - старый вариант "<task_id>.jpg"
    """
    folder = app.config.get('UPLOAD_FOLDER', 'images')
    if not os.path.isdir(folder):
        return []

    patterns = [
        os.path.join(folder, f"{task_id}_*.*"),
        os.path.join(folder, f"{task_id}.*")
    ]
    found = []
    for pat in patterns:
        for p in glob.glob(pat):
            if os.path.isfile(p):
                found.append(os.path.basename(p))

    # убираем дубликаты, сортируем: сначала старый "<id>.<ext>", затем "<id>_1", "<id>_2"
    uniq = list(dict.fromkeys(found))
    def sort_key(fname):
        m = re.match(rf'^{re.escape(str(task_id))}_(\d+)\.(.+)$', fname)
        if m:
            return (1, int(m.group(1)), fname)
        m2 = re.match(rf'^{re.escape(str(task_id))}\.(.+)$', fname)
        if m2:
            return (0, 0, fname)
        return (2, 0, fname)
    uniq.sort(key=sort_key)
    return uniq

def clean_option_line(line):
    """
    Очищает строку варианта от маркеров (o, •, буквы с точкой, галочки) и
    возвращает (cleaned_text, is_correct_flag).
    Возвращает (None, False) для строк, которые следует пропустить (например, номер страницы).
    """
    correct_mark_chars = ['','✓','✔','✗','☑','*']
    is_correct = any(ch in line for ch in correct_mark_chars)
    # удаляем явные символы галочек
    for ch in correct_mark_chars:
        line = line.replace(ch, '')
    # удаляем ведущие bullets/символы
    line = re.sub(r'^[\s\-\–\—\*\u2022\u25CF\u25CBoО0\(\)\[\]]+', '', line)
    # удаляем маркировку "A." "B)" "1." и т.п.
    line = re.sub(r'^[A-Za-zА-Яа-я]\s*[\.\)]\s*', '', line)
    line = re.sub(r'^\d{1,3}\s*[\.\)]\s*', '', line)
    line = line.strip()
    # пропускать чисто цифровые строки (чаще всего — номера страниц)
    if re.match(r'^\d{1,3}$', line):
        return None, False
    if not line:
        return None, False
    return line, is_correct

def get_tasks_by_ids(task_ids):
    if not task_ids: return []
    placeholders = ','.join('?' * len(task_ids))
    query = f"SELECT * FROM tasks6 WHERE id IN ({placeholders})"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, task_ids)
    tasks = cursor.fetchall()
    conn.close()
    return tasks

def create_attempt(test_id, user_id, score, answers):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attempts (test_id, user_id, score, answers, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (test_id, user_id, score, str(answers), datetime.now().isoformat()))
    conn.commit()
    attempt_id = cursor.lastrowid
    conn.close()
    return attempt_id

def get_user_attempts(test_id, user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attempts WHERE test_id = ? AND user_id = ? ORDER BY timestamp DESC", (test_id, user_id))
    attempts = cursor.fetchall()
    conn.close()
    return attempts

def get_attempt_by_id(attempt_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
    attempt = cursor.fetchone()
    conn.close()
    return attempt

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, name TEXT, surname TEXT, password TEXT NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS tasks6 (id INTEGER PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL, anwser TEXT, difficulty INTEGER, tags TEXT, source TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS tests (id INTEGER PRIMARY KEY AUTOINCREMENT, time INTEGER NOT NULL, attempts INTEGER NOT NULL, tasks_id TEXT NOT NULL, access_code TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER NOT NULL, user_id INTEGER NOT NULL, score INTEGER NOT NULL, answers TEXT NOT NULL, timestamp TEXT NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS teachers (user_id INTEGER PRIMARY KEY, FOREIGN KEY(user_id) REFERENCES users(id))")
    cursor.execute("CREATE TABLE IF NOT EXISTS teacher_students (teacher_id INTEGER NOT NULL, student_id INTEGER NOT NULL, FOREIGN KEY(teacher_id) REFERENCES users(id), FOREIGN KEY(student_id) REFERENCES users(id), PRIMARY KEY (teacher_id, student_id))")
    conn.commit()
    conn.close()

def create_test(time, attempts, tasks_id, access_code=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tests (time, attempts, tasks_id, access_code) VALUES (?, ?, ?, ?)", (time, attempts, tasks_id, access_code))
    conn.commit()
    test_id = cursor.lastrowid
    conn.close()
    return test_id

def get_all_tests():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tests")
    tests = cursor.fetchall()
    conn.close()
    return tests

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

def register_user(username, password, name, surname):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, name, surname, password) VALUES (?, ?, ?, ?)", (username, name, surname, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    
def process_pdf_and_create_tasks_clean(pdf_path: str, uploader_user_id: int, db_path: str, images_output_dir="images"):
    """
    Разбирает PDF и сохраняет задания в БД tasks6 + изображения в папку images/.
    Возвращает список словарей с созданными заданиями.
    """
    parsed_tasks = parse_pdf_to_tasks_clean(pdf_path, images_output_dir=images_output_dir)

    import sqlite3, os
    from pathlib import Path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    created = []
    for t in parsed_tasks:
        title = f"Задание {t['number']}" if t.get('number') else (t['question'][:50] if t['question'] else "Задание")
        description = t['question']
        answer = ""  # пока ответов нет — только варианты
        difficulty = 1
        tags = ""
        source = os.path.basename(pdf_path)

        cursor.execute(
            "INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source) VALUES (?, ?, ?, ?, ?, ?)",
            (title, description, answer, difficulty, tags, source)
        )
        task_id = cursor.lastrowid

        # сохранить картинки с привязкой к task_id
        saved_images = []
        for idx, imgfile in enumerate(t['images'], start=1):
            img_src = os.path.join(images_output_dir, imgfile)
            if os.path.exists(img_src):
                ext = os.path.splitext(img_src)[1].lstrip(".") or "jpg"
                dest = os.path.join(images_output_dir, f"{task_id}_{idx}.{ext}")
                os.replace(img_src, dest)
                saved_images.append(os.path.basename(dest))

        created.append({
            "task_id": task_id,
            "number": t.get("number"),
            "title": title,
            "description": description,
            "answers": t.get("answers", []),
            "correct": t.get("correct", []),
            "images": saved_images,
        })

    conn.commit()
    conn.close()
    return created


def process_pdf_and_create_tasks(pdf_path: str, uploader_user_id: int, db_path: str):
    """
    Извлекает текст и изображения из PDF, создаёт записи задач в БД и связывает изображения с ближайшими по положению номерами заданий.
    Сохранение изображений: app.config['UPLOAD_FOLDER']/<task_id>_<index>.<ext>
    Возвращает список созданных записей вида:
      [{'task_id': id, 'task_num': num, 'title': title, 'description': desc, 'images': [paths...]}, ...]
    """
    results = []
    doc = fitz.open(pdf_path)

    # 1) Собираем текст по страницам
    pages_text = [doc[p].get_text("text") or "" for p in range(len(doc))]
    full_text = "\n".join(pages_text)

    # 2) Найдём в тексте потенциальные номера заданий — несколько шаблонов для надёжности
    # Поддерживаем: "12.", "12)", "№12", "Задание 12", "12 —", "12 — " и т.д.
    anchor_patterns = [
        r'(?m)^\s*(\d{1,3})\.\s+',       # 12.
        r'(?m)^\s*(\d{1,3})\)\s+',       # 12)
        r'(?m)^\s*№\s*(\d{1,3})\s+',     # №12
        r'(?m)^\s*Задание\s+(\d{1,3})',  # Задание 12
        r'(?m)^\s*(\d{1,3})\s*—\s+'      # 12 —
    ]
    anchors_from_text = []  # (page_idx, y_center, task_num, x_center)
    # We'll also detect anchors by scanning page words to get exact positions
    for p_idx in range(len(doc)):
        words = doc[p_idx].get_text("words")  # list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        for w in words:
            token = w[4].strip()
            # simple checks: token like "12." or "12)" or "№12" or "12—"
            m = re.match(r'^(?:№\s*)?(\d{1,3})([.)—]?)$', token)
            if m:
                num = int(m.group(1))
                x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
                x_center = (x0 + x1) / 2.0
                y_center = (y0 + y1) / 2.0
                anchors_from_text.append({'page': p_idx, 'y': y_center, 'x': x_center, 'num': num})
        # also attempt to match full lines for patterns like "Задание 12"
        blocks = doc[p_idx].get_text("blocks")
        for b in blocks:
            block_text = (b[4] or "").strip()
            for pat in anchor_patterns:
                for mm in re.finditer(pat, block_text):
                    try:
                        num = int(mm.group(1))
                    except Exception:
                        continue
                    # approximate y by block bbox
                    bbox = b[0:4]  # (x0, y0, x1, y1)
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    y_center = (bbox[1] + bbox[3]) / 2.0
                    anchors_from_text.append({'page': p_idx, 'y': y_center, 'x': x_center, 'num': num})
    # Deduplicate anchors (same page + very close y + same num)
    anchors = []
    for a in anchors_from_text:
        dup = False
        for ex in anchors:
            if ex['num'] == a['num'] and ex['page'] == a['page'] and abs(ex['y'] - a['y']) < 3.0:
                dup = True
                break
        if not dup:
            anchors.append(a)
    # If no anchors found by words, fallback to scanning entire text (less precise)
    if not anchors:
        for m in re.finditer(r'(?m)^\s*(\d{1,3})\.\s*', full_text):
            # can't get exact page/pos here; we will still create tasks but anchors empty
            anchors.append({'page': 0, 'y': 0.0, 'x': 0.0, 'num': int(m.group(1))})

    # 3) Извлекаем изображения и их bbox-ы (сохраним временно в папке)
    images_root_temp = os.path.join(app.config['UPLOAD_FOLDER'], "pdf_tmp_images")
    Path(images_root_temp).mkdir(parents=True, exist_ok=True)
    extracted_images = []  # items: {'temp_path':..., 'page':p_idx, 'bbox':(x0,y0,x1,y1), 'ext': ext}
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:  # image block
                bbox = block.get("bbox", [0,0,0,0])
                img_obj = block.get("image")
                image_bytes = None
                ext = None
                xref = None
                if isinstance(img_obj, dict):
                    xref = img_obj.get("xref")
                    if xref is not None:
                        try:
                            base = doc.extract_image(xref)
                            image_bytes = base.get("image")
                            ext = base.get("ext")
                        except Exception:
                            image_bytes = None
                    else:
                        possible = img_obj.get("image")
                        if isinstance(possible, (bytes, bytearray)):
                            image_bytes = bytes(possible)
                elif isinstance(img_obj, (bytes, bytearray)):
                    image_bytes = bytes(img_obj)
                elif isinstance(img_obj, int):
                    xref = img_obj
                    try:
                        base = doc.extract_image(xref)
                        image_bytes = base.get("image")
                        ext = base.get("ext")
                    except Exception:
                        image_bytes = None
                if not image_bytes:
                    continue
                if not ext:
                    ext = guess_ext_from_bytes(image_bytes)
                # временное имя
                temp_base = f"p{p_idx+1}_img_{len(extracted_images)+1}"
                temp_path = save_image_bytes(image_bytes, images_root_temp, temp_base, ext)
                extracted_images.append({'temp_path': temp_path, 'page': p_idx, 'bbox': tuple(bbox), 'ext': ext})

    # 4) Разбиваем текст на блоки задач так же, как раньше — чтобы создать записи в БД
    # Ищем начала каждого задания в полном тексте (позиции), затем режем на блоки.
    pattern = re.compile(r'(?m)^\s*(\d{1,3})\.\s*')
    starts = [(m.start(), int(m.group(1))) for m in pattern.finditer(full_text)]
    tasks_text_blocks = []
    if not starts:
        # если не нашли глобально — пробуем постранично
        cursor = 0
        for p_idx, txt in enumerate(pages_text):
            for m in re.finditer(r'(?m)^\s*(\d{1,3})\.\s*', txt):
                starts.append((cursor + m.start(), int(m.group(1))))
            cursor += len(txt) + 1
    if starts:
        for i, (pos, num) in enumerate(starts):
            start_pos = pos
            end_pos = starts[i+1][0] if i + 1 < len(starts) else len(full_text)
            block = full_text[start_pos:end_pos].strip()
            block = re.sub(r'^\s*\d{1,3}\.\s*', '', block, count=1, flags=re.M)
            tasks_text_blocks.append((num, block))
    else:
        # если вообще ничего не найдено — создаём одну задачу с полным текстом
        tasks_text_blocks.append((None, full_text))

    # 5) Создаём задачи в БД (без изображений) и подготовим карту номер->task_id
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    created = []
    for (num, block_text) in tasks_text_blocks:
        title = f"Задание {num}" if num is not None else (block_text.splitlines()[0][:120] if block_text else "Задание")
        description = block_text
        answer = ""
        difficulty = 1
        tags = ""
        source = os.path.basename(pdf_path)
        cursor.execute("INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source) VALUES (?, ?, ?, ?, ?, ?)",
                       (title, description, answer, difficulty, tags, source))
        task_id = cursor.lastrowid
        created.append({'task_id': task_id, 'task_num': num, 'title': title, 'description': description, 'images': []})
    conn.commit()

    num_to_task_id = {c['task_num']: c['task_id'] for c in created if c['task_num'] is not None}
    # Also build list of anchors mapped to actual task_id when possible
    anchor_objs = []
    for a in anchors:
        num = a['num']
        task_id = num_to_task_id.get(num)
        anchor_objs.append({'task_num': num, 'task_id': task_id, 'page': a['page'], 'y': a['y'], 'x': a['x']})

    # If no anchors had a task_id (e.g. anchors parsed but numbers didn't match created blocks),
    # then construct anchor_objs from created list by estimating page=0,y=0 fallback.
    if not any(a.get('task_id') for a in anchor_objs):
        anchor_objs = []
        idx = 0
        # simple fallback: spread tasks across pages evenly
        total_pages = max(1, len(doc))
        for c in created:
            approx_page = min(len(doc)-1, idx * total_pages // max(1, len(created)))
            anchor_objs.append({'task_num': c['task_num'], 'task_id': c['task_id'], 'page': approx_page, 'y': 100.0 + idx*10, 'x': 50.0})
            idx += 1

    # 6) Функция оценки соответствия изображения -> анктора (меньше лучше)
    def score_match(img_page, img_mid_y, anchor):
        # page difference is the strongest signal
        page_diff = abs(img_page - anchor['page'])
        # vertical distance
        vdist = abs(img_mid_y - anchor['y'])
        score = page_diff * 10000 + vdist
        # небольшой штраф если изображение расположено намного выше анктора (возможно оно относится к предыдущему анктора)
        if img_mid_y + 20 < anchor['y']:
            score += 2000
        return score

    # 7) Для каждого изображения подбираем лучший анкор
    for img in extracted_images:
        bbox = img['bbox']
        img_page = img['page']
        img_mid_y = (bbox[1] + bbox[3]) / 2.0
        best_anchor = None
        best_score = None
        for anchor in anchor_objs:
            sc = score_match(img_page, img_mid_y, anchor)
            if best_score is None or sc < best_score:
                best_score = sc
                best_anchor = anchor
        # Если лучший анкор найден — привязываем изображение к соответствующему task_id
        if best_anchor and best_anchor.get('task_id'):
            tid = best_anchor['task_id']
            # сохраняем изображение в корне UPLOAD_FOLDER как <taskid>_N.<ext>
            # найдем текущую позицию для нумерации
            existing = [p for c in created if c['task_id'] == tid for p in c['images']]
            next_index = 1 + sum(1 for p in existing)
            dest_filename = f"{tid}_{next_index}.{img['ext'].lstrip('.')}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], dest_filename)
            try:
                # перемещаем temp -> dest (перезаписывать не нужно, имя уникально)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.replace(img['temp_path'], dest_path)
            except Exception:
                # fallback копирование
                try:
                    with open(img['temp_path'], 'rb') as rf, open(dest_path, 'wb') as wf:
                        wf.write(rf.read())
                    os.remove(img['temp_path'])
                except Exception:
                    continue
            # обновляем созданный объект
            for c in created:
                if c['task_id'] == tid:
                    c['images'].append(dest_path)
                    break
            # если в таблице tasks6 есть колонка image_path — записать первый путь туда
            try:
                cursor.execute("PRAGMA table_info('tasks6')")
                cols = [cinfo[1] for cinfo in cursor.fetchall()]
                if 'image_path' in cols:
                    # если поле пустое — заполнить первым изображением
                    cursor.execute("SELECT image_path FROM tasks6 WHERE id = ?", (tid,))
                    cur_val = cursor.fetchone()[0]
                    if not cur_val:
                        cursor.execute("UPDATE tasks6 SET image_path = ? WHERE id = ?", (dest_path, tid))
                        conn.commit()
            except Exception:
                conn.rollback()
        else:
            # Нет подходящего анктора — оставляем временный файл в tmp или удаляем его
            try:
                os.remove(img['temp_path'])
            except Exception:
                pass

    # 8) Очистка временной папки (если остались файлы)
    try:
        if os.path.isdir(images_root_temp):
            for f in os.listdir(images_root_temp):
                fp = os.path.join(images_root_temp, f)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass
            try:
                os.rmdir(images_root_temp)
            except Exception:
                pass
    except Exception:
        pass

    conn.close()
    return created



@app.route('/add-student', methods=['POST'])
def add_student():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = request.form['username']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        flash('Только учителя могут добавлять учеников', 'error')
        return redirect(url_for('home'))
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    student = cursor.fetchone()
    if not student:
        conn.close()
        flash('Пользователь не найден', 'error')
        return redirect(url_for('teacher_dashboard'))
    student_id = student[0]
    cursor.execute("SELECT 1 FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (session['user_id'], student_id))
    if cursor.fetchone():
        conn.close()
        flash('Ученик уже добавлен', 'info')
        return redirect(url_for('teacher_dashboard'))
    cursor.execute("INSERT INTO teacher_students (teacher_id, student_id) VALUES (?, ?)", (session['user_id'], student_id))
    conn.commit()
    conn.close()
    flash('Ученик добавлен', 'success')
    return redirect(url_for('teacher_dashboard'))

def get_teacher_students(teacher_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT users.id, users.name, users.surname, users.username FROM teacher_students JOIN users ON teacher_students.student_id = users.id WHERE teacher_students.teacher_id = ?", (teacher_id,))
    students = [{'id': row[0], 'name': row[1], 'surname': row[2], 'username': row[3]} for row in cursor.fetchall()]
    conn.close()
    return students

def get_student_attempts(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT a.id, a.test_id, a.score, a.timestamp, t.time as duration FROM attempts a JOIN tests t ON a.test_id = t.id WHERE a.user_id = ? ORDER BY a.timestamp DESC", (student_id,))
    attempts = [{'id': row[0], 'test_id': row[1], 'score': row[2], 'timestamp': datetime.fromisoformat(row[3]), 'duration': row[4]} for row in cursor.fetchall()]
    conn.close()
    return attempts

def delete_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM teacher_students WHERE teacher_id = ? OR student_id = ?", (user_id, user_id))
        cursor.execute("DELETE FROM teachers WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM attempts WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        current_app.logger.error(f"Error deleting user: {e}")
        return False
    finally:
        conn.close()

def delete_task(task_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
        cursor.execute("DELETE FROM tasks6 WHERE id = ?", (task_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        current_app.logger.error(f"Error deleting task: {e}")
        return False
    finally:
        conn.close()

def delete_test(test_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM attempts WHERE test_id = ?", (test_id,))
        cursor.execute("DELETE FROM tests WHERE id = ?", (test_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        current_app.logger.error(f"Error deleting test: {e}")
        return False
    finally:
        conn.close()

def get_student_stats(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE user_id = ?", (student_id,))
    total_tests = cursor.fetchone()[0] or 0
    cursor.execute("SELECT AVG(score) FROM attempts WHERE user_id = ?", (student_id,))
    average_score = round(cursor.fetchone()[0] or 0, 1)
    cursor.execute("SELECT MAX(score) FROM attempts WHERE user_id = ?", (student_id,))
    best_score = cursor.fetchone()[0] or 0
    conn.close()
    return {'total_tests': total_tests, 'average_score': average_score, 'best_score': best_score}

def is_admin():
    return 'user_info' in session and session['user_info'].get('username') == 'admin'

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users")
    users = [{'id': row[0], 'username': row[1], 'name': row[2], 'surname': row[3], 'is_teacher': is_user_teacher(row[0])} for row in cursor.fetchall()]
    conn.close()
    return users

def is_user_teacher(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (user_id,))
    result = cursor.fetchone() is not None
    conn.close()
    return result

def toggle_teacher_role(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if is_user_teacher(user_id):
        cursor.execute("DELETE FROM teachers WHERE user_id = ?", (user_id,))
        action = "removed"
    else:
        cursor.execute("INSERT INTO teachers (user_id) VALUES (?)", (user_id,))
        action = "added"
    conn.commit()
    conn.close()
    return action

@app.route('/admin')
def admin_panel():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    users = get_all_users()
    return render_template('admin_panel.html', users=users, user=session.get('user_info'))

@app.route('/admin/toggle-teacher/<int:user_id>')
def admin_toggle_teacher(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    action = toggle_teacher_role(user_id)
    flash('Роль учителя обновлена', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/view-user/<int:user_id>')
def admin_view_user(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('admin_panel'))
    user_info = {'id': user[0], 'username': user[1], 'name': user[2], 'surname': user[3], 'is_teacher': is_user_teacher(user_id)}
    students = []
    if user_info['is_teacher']:
        cursor.execute("SELECT u.id, u.username, u.name, u.surname FROM teacher_students ts JOIN users u ON ts.student_id = u.id WHERE ts.teacher_id = ?", (user_id,))
        students = [{'id': row[0], 'username': row[1], 'name': row[2], 'surname': row[3]} for row in cursor.fetchall()]
    cursor.execute("SELECT a.id, a.test_id, a.score, a.timestamp, t.time FROM attempts a JOIN tests t ON a.test_id = t.id WHERE a.user_id = ? ORDER BY a.timestamp DESC", (user_id,))
    attempts = [{'id': row[0], 'test_id': row[1], 'score': row[2], 'timestamp': row[3], 'time': row[4]} for row in cursor.fetchall()]
    conn.close()
    return render_template('admin_user_view.html', user_info=user_info, students=students, attempts=attempts, current_user=session.get('user_info'))

@app.route('/admin/add-student', methods=['POST'])
def admin_add_student():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    teacher_id = request.form.get('teacher_id')
    student_username = request.form.get('student_username')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (teacher_id,))
    if not cursor.fetchone():
        flash('Пользователь не учитель', 'error')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    cursor.execute("SELECT id FROM users WHERE username = ?", (student_username,))
    student = cursor.fetchone()
    if not student:
        flash('Ученик не найден', 'error')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    student_id = student[0]
    cursor.execute("SELECT 1 FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (teacher_id, student_id))
    if cursor.fetchone():
        flash('Ученик уже добавлен', 'info')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    cursor.execute("INSERT INTO teacher_students (teacher_id, student_id) VALUES (?, ?)", (teacher_id, student_id))
    conn.commit()
    conn.close()
    flash('Ученик добавлен', 'success')
    return redirect(url_for('admin_view_user', user_id=teacher_id))

@app.route('/admin/remove-student', methods=['POST'])
def admin_remove_student():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    teacher_id = request.form.get('teacher_id')
    student_id = request.form.get('student_id')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (teacher_id, student_id))
    conn.commit()
    conn.close()
    flash('Ученик удален', 'success')
    return redirect(url_for('admin_view_user', user_id=teacher_id))

@app.route('/teacher/student/<int:student_id>')
def view_student_results(student_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    cursor.execute("SELECT 1 FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (session['user_id'], student_id))
    has_access = cursor.fetchone() is not None
    conn.close()
    if not is_teacher or not has_access:
        flash('Нет доступа', 'error')
        return redirect(url_for('home'))
    student = fetch_user(student_id)
    if not student:
        flash('Ученик не найден', 'error')
        return redirect(url_for('teacher_dashboard'))
    attempts = get_student_attempts(student_id)
    stats = get_student_stats(student_id)
    return render_template('student_results.html', student=student, attempts=attempts, stats=stats)

@app.route('/get-test-html/<int:test_id>')
def get_test_html(test_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        return "Доступ запрещен", 403
    test = get_test_by_id(test_id)
    if not test:
        return "Тест не найден", 404
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    return render_template('test_pdf.html', test=test, tasks=tasks, get_difficulty_color=get_difficulty_color)

@app.route('/remove-student/<int:student_id>', methods=['POST'])
def remove_student(student_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        flash('Только учителя могут удалять учеников', 'error')
        return redirect(url_for('home'))
    cursor.execute("DELETE FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (session['user_id'], student_id))
    conn.commit()
    conn.close()
    flash('Ученик удален', 'success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/create-task', methods=['GET', 'POST'])
def create_task():
    """
    Создание задания — теперь поддерживает несколько изображений.
    Файлы можно передавать:
      - одиночный input с name="image"
      - множественный input с name="images" и attribute multiple
    Сохранение: app.config['UPLOAD_FOLDER']/<task_id>_1.<ext>, <task_id>_2.<ext>, ...
    Если в таблице tasks6 есть колонка image_path — в неё записывается путь первого изображения.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        flash('Только учителя могут создавать тесты', 'error')
        return redirect(url_for('home'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        answer = request.form.get('answer', '').strip()
        difficulty = request.form.get('difficulty', '').strip()
        tags = request.form.get('tags', '').strip()
        source = request.form.get('source', '').strip()
        # простая валидация
        if not title or not description or not answer or not difficulty:
            flash('Заполните обязательные поля', 'error')
            conn.close()
            return render_template('create_task.html', user=session.get('user_info'))
        try:
            difficulty_int = int(difficulty)
            if not 1 <= difficulty_int <= 10:
                raise ValueError
        except ValueError:
            flash('Сложность от 1 до 10', 'error')
            conn.close()
            return render_template('create_task.html', user=session.get('user_info'))
        # создаём запись задачи
        cursor.execute("INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source) VALUES (?, ?, ?, ?, ?, ?)",
                       (title, description, answer, difficulty_int, tags, source))
        task_id = cursor.lastrowid

        # собираем файлы — поддерживаем оба варианта
        uploaded = []
        # множественный input
        uploaded += request.files.getlist('images') if 'images' in request.files else []
        # одиночный input
        single = request.files.get('image')
        if single and single.filename:
            uploaded.append(single)

        # Сохраняем файлы как <task_id>_1.ext, <task_id>_2.ext, ...
        saved_any = False
        idx = 1
        for f in uploaded:
            if not f or not f.filename:
                continue
            if not allowed_image(f.filename):
                # пропускаем недопустимые форматы (или можно flash-ить)
                continue
            filename = secure_filename(f.filename)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
            dest_name = f"{task_id}_{idx}.{ext}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], dest_name)
            try:
                # перезапись не нужна, имя уникально по индексу
                f.save(dest_path)
                saved_any = True
                idx += 1
            except Exception:
                # игнорируем неудачные сохранения отдельных файлов
                continue

        # Если есть колонка image_path, установим её в первый найденный файл
        try:
            cursor.execute("PRAGMA table_info('tasks6')")
            cols = [c[1] for c in cursor.fetchall()]
            if 'image_path' in cols:
                # найдём первый файл (по	get_task_images)
                imgs = get_task_images(task_id)
                if imgs:
                    first = os.path.join(app.config['UPLOAD_FOLDER'], imgs[0])
                    cursor.execute("UPDATE tasks6 SET image_path = ? WHERE id = ?", (first, task_id))
                    conn.commit()
        except Exception:
            conn.rollback()

        conn.commit()
        conn.close()
        flash('Задание создано', 'success')
        return redirect(url_for('home'))
    conn.close()
    return render_template('create_task.html', user=session.get('user_info'))


@app.route('/check-time')
def check_time():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    test_id = request.args.get('test_id')
    if not test_id:
        return jsonify({'error': 'Test ID required'}), 400
    test = get_test_by_id(test_id)
    if not test:
        return jsonify({'error': 'Test not found'}), 404
    start_time = session.get(f'test_{test_id}_start_time')
    if not start_time:
        return jsonify({'error': 'Test not started'}), 400
    elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
    time_left = max(0, test[1] * 60 - elapsed)
    return jsonify({'time_left': time_left, 'test_duration': test[1] * 60})

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session or session['user_id'] == 0:
        flash('Войдите в систему', 'error')
        return redirect(url_for('login'))
    user_id = session['user_id']
    user = fetch_user(user_id)
    if not user:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        new_name = request.form.get('name', '').strip()
        new_surname = request.form.get('surname', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        updates = {}
        if new_name and new_name != (user.get('name') or ''):
            updates['name'] = new_name
        if new_surname and new_surname != (user.get('surname') or ''):
            updates['surname'] = new_surname
        if new_password:
            if not current_password:
                flash('Укажите текущий пароль', 'error')
                return redirect(url_for('profile'))
            stored = user.get('password') or ''
            pw_ok = check_password_hash(stored, current_password) if stored.startswith('pbkdf2:sha256:') else stored == current_password
            if not pw_ok:
                flash('Текущий пароль неверен', 'error')
                return redirect(url_for('profile'))
            if new_password != confirm_password:
                flash('Пароли не совпадают', 'error')
                return redirect(url_for('profile'))
            if len(new_password) < 6:
                flash('Пароль минимум 6 символов', 'error')
                return redirect(url_for('profile'))
            updates['password'] = generate_password_hash(new_password)
        if not updates:
            flash('Нечего обновлять', 'info')
            return redirect(url_for('profile'))
        set_clause = ", ".join([f"{k} = ?" for k in updates])
        params = list(updates.values()) + [user_id]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {set_clause} WHERE id = ?", params)
        conn.commit()
        conn.close()
        session['user_info'] = fetch_user(user_id)
        flash('Профиль обновлён', 'success')
        return redirect(url_for('profile'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT attempts.id, attempts.test_id, attempts.score, attempts.timestamp, tests.time, tests.attempts FROM attempts JOIN tests ON attempts.test_id = tests.id WHERE attempts.user_id = ? ORDER BY attempts.timestamp DESC", (user_id,))
    attempts = cursor.fetchall()
    conn.close()
    user_info = session.get('user_info') or {'id': user['id'], 'username': user['username'], 'name': user['name'], 'surname': user['surname']}
    return render_template('profile.html', attempts=attempts, user=user_info)

@app.route('/test/<int:test_id>', methods=['GET', 'POST'])
def take_test(test_id):
    if 'user_id' not in session:
        flash('Войдите в систему', 'error')
        return redirect(url_for('login'))
    test = get_test_by_id(test_id)
    if not test:
        flash('Тест не найден', 'error')
        return redirect(url_for('view_tests'))
    if test[4] and not session.get(f'test_{test_id}_access'):
        return redirect(url_for('start_test', test_id=test_id))
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    if not tasks:
        flash('Задания не найдены', 'error')
        return redirect(url_for('view_tests'))
    user_attempts = get_user_attempts(test_id, session['user_id'])
    attempts_left = test[2] - len(user_attempts)
    if attempts_left <= 0 and session['user_id'] != 0:
        flash('Нет попыток', 'error')
        return redirect(url_for('view_tests'))
    if request.method == 'POST':
        user_answers = {}
        correct_count = 0
        for task in tasks:
            user_answer = request.form.get(f"answer_{task[0]}", '').strip()
            user_answers[task[0]] = user_answer
            if user_answer.lower() == (task[3] or '').lower():
                correct_count += 1
        score = int((correct_count / len(tasks)) * 100) if tasks else 0
        attempt_id = create_attempt(test_id, session['user_id'], score, user_answers)
        return redirect(url_for('test_result', attempt_id=attempt_id))
    session[f'test_{test_id}_start_time'] = datetime.now().isoformat()
    return render_template('take_test.html', test=test, tasks=tasks, get_difficulty_color=get_difficulty_color, attempts_left=attempts_left, user=session.get('user_info'))

@app.route('/test/result/<int:attempt_id>')
def test_result(attempt_id):
    attempt = get_attempt_by_id(attempt_id)
    if not attempt:
        flash('Результат не найден', 'error')
        return redirect(url_for('view_tests'))
    if attempt[2] != session['user_id']:
        flash('Нет доступа', 'error')
        return redirect(url_for('view_tests'))
    test = get_test_by_id(attempt[1])
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    import ast
    user_answers = ast.literal_eval(attempt[4])
    return render_template('test_result.html', test=test, attempt=attempt, tasks=tasks, user_answers=user_answers, get_difficulty_color=get_difficulty_color, user=session.get('user_info'))

@app.route('/create-test', methods=['GET', 'POST'])
def create_test_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        flash('Только учителя могут создавать тесты', 'error')
        return redirect(url_for('home'))
    if request.method == 'POST':
        time = request.form.get('time')
        attempts = request.form.get('attempts')
        tasks_id = request.form.get('tasks_id')
        access_code = request.form.get('access_code', None)
        if not time or not attempts or not tasks_id:
            flash('Заполните поля', 'error')
        else:
            try:
                create_test(int(time), int(attempts), tasks_id, access_code)
                return redirect(url_for('home'))
            except ValueError:
                flash('Некорректные значения', 'error')
    conn.close()
    return render_template('create_test.html', user=session.get('user_info'))

@app.route('/start-test/<int:test_id>', methods=['GET', 'POST'])
def start_test(test_id):
    if 'user_id' not in session:
        flash('Войдите в систему', 'error')
        return redirect(url_for('login'))
    test = get_test_by_id(test_id)
    if not test:
        flash('Тест не найден', 'error')
        return redirect(url_for('view_tests'))
    if test[4]:
        if request.method == 'POST':
            entered_code = request.form.get('access_code', '').strip()
            if entered_code == test[4]:
                session[f'test_{test_id}_access'] = True
                return redirect(url_for('take_test', test_id=test_id))
        return render_template('access_code.html', test=test)
    return redirect(url_for('take_test', test_id=test_id))

@app.route('/tests')
def view_tests():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    tests = get_all_tests()
    return render_template('tests.html', tests=tests, user=session.get('user_info'))

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    conn.close()
    return render_template('index.html', user=session.get('user_info'), is_teacher=is_teacher)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['user_info'] = {'username': user[1], 'name': user[2], 'surname': user[3]}
            return redirect(url_for('home'))
        flash('Неверный логин или пароль', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form.get('name', '')
        surname = request.form.get('surname', '')
        if register_user(username, password, name, surname):
            flash('Регистрация успешна', 'success')
            return redirect(url_for('login'))
        flash('Логин занят', 'error')
    return render_template('register.html')

@app.route('/guest')
def guest():
    session['user_id'] = 0
    session['user_info'] = {'username': 'Гость'}
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/upload-pdf", methods=["GET", "POST"])
def upload_pdf():
    if 'user_id' not in session:
        return jsonify({'status': 'unauthenticated', 'message': 'login required'}) \
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' \
            else redirect(url_for('login'))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'forbidden', 'message': 'Только учителя'})
        flash("Только учителя могут загружать PDF", "error")
        return redirect(url_for('home'))
    conn.close()

    if request.method == "POST":
        file = request.files.get('file')
        if not file:
            msg = "Файл не найден"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': msg})
            flash(msg, "error")
            return redirect(request.url)

        filename = file.filename
        if not allowed_file(filename):
            msg = "Только PDF"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': msg})
            flash(msg, "error")
            return redirect(request.url)

        tmp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(tmp_dir, secure_filename(filename) or "uploaded.pdf")
        file.save(pdf_path)

        created = parse_pdf_and_store_tasks(pdf_path, DB_PATH, images_output_dir="images")

        os.remove(pdf_path)
        os.rmdir(tmp_dir)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'ok', 'created': [
                {'task_id': c['task_id'], 'task_num': c.get('number')} for c in created
            ]})
        flash(f"Создано {len(created)} заданий", "success")
        return redirect(url_for('home'))

    return render_template("upload_pdf.html")



@app.route('/print-test/<int:test_id>')
def print_test(test_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        return "Доступ запрещен", 403
    test = get_test_by_id(test_id)
    if not test:
        conn.close()
        return "Тест не найден", 404
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    conn.close()
    return render_template('print_test.html', test=test, tasks=tasks, user=session.get('user_info'))

@app.route('/search', methods=['GET'])
def search():
    """
    Обновлённый маршрут /search с поддержкой пагинации.
    - собирает параметры поиска
    - вызывает search_tasks(...) который возвращает словарь с задачами и информацией о пагинации
    - преобразует каждую строку в dict с полями, которые ожидает tasks.html
    - добавляет список изображений (task['images'] = [{'filename': '...'}, ...])
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # собираем параметры поиска (как у вас было)
    search_params = {k: request.args.get(k, '').strip() for k in ['title', 'description', 'tag', 'source', 'min_difficulty', 'max_difficulty', 'task_id', 'min_task_id', 'max_task_id']}
    
    # получаем номер страницы из параметров
    page = int(request.args.get('page', 1))
    if page < 1:
        page = 1

    # получаем данные с пагинацией
    search_result = search_tasks(search_params, page=page, per_page=20)
    raw_tasks = search_result['tasks']

    # определяем, является ли пользователь учителем (как раньше)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cur.fetchone() is not None
    conn.close()

    # Преобразуем raw_tasks (tuple rows) в список словарей, совместимых с tasks.html
    tasks = []
    for row in raw_tasks:
        # Ожидаем формат: (rowid, title, description, anwser, difficulty, tags, source)
        # Но будем аккуратно: если длина другая — попытаемся извлечь по индексам безопасно.
        try:
            task_id = row[0]
            title = row[1] if len(row) > 1 else ''
            description = row[2] if len(row) > 2 else ''
            answer = row[3] if len(row) > 3 else ''
            difficulty = row[4] if len(row) > 4 else 1
            tags = row[5] if len(row) > 5 else ''
            source = row[6] if len(row) > 6 else ''
        except Exception:
            # если неожиданно row не индексируемый — пропускаем
            continue

        # Получаем списoк имён файлов, относящихся к задаче (get_task_images должен возвращать список имён файлов)
        try:
            image_filenames = get_task_images(task_id)  # возвращает список, например ["89_1.jpg","89_2.png"]
        except Exception:
            image_filenames = []

        # Преобразуем в объекты, которые шаблон сможет адресовать через .filename
        images = [{'filename': fn} for fn in image_filenames]

        tasks.append({
            'id': task_id,
            'title': title,
            'description': description,
            'answer': answer,
            'difficulty': difficulty or 1,
            'tags': tags,
            'source': source,
            'images': images
        })

    # Передаём в шаблон
    return render_template(
        'tasks.html',
        tasks=tasks,
        get_difficulty_color=get_difficulty_color,
        search_params=search_params,
        user=session.get('user_info'),
        is_teacher=is_teacher,
        pagination=search_result
    )

@app.route('/edit-task/<int:task_id>', methods=['GET', 'POST'])
def edit_task(task_id):
    """
    Редактирование задания — поддерживает множественные изображения.
    Ожидает в форме:
      - файлы: input name="images" (multiple) или одиночный name="image"
      - удаление: checkboxes name="delete_images" (значение = имя файла, например "89_1.jpg")
        также поддерживается старое поле delete_image (checkbox) — в этом случае будет удалён путь из image_path (если он есть).
    После сохранения поле image_path в БД обновляется на первый доступный файл или NULL.
    """
    if 'user_id' not in session:
        flash("Авторизация требуется", "error")
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cur.fetchone():
        conn.close()
        flash("Только учителя могут редактировать", "error")
        return redirect(url_for('home'))

    # Получаем строку задачи
    cur.execute("SELECT rowid, * FROM tasks6 WHERE rowid = ?", (task_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        abort(404)
    # Получаем колонки и преобразуем в dict
    cur.execute("PRAGMA table_info('tasks6')")
    cols = [c[1] for c in cur.fetchall()]
    task = {cols[i]: row[i+1] for i in range(len(cols))}
    task['id'] = task_id

    # текущие изображения (имена файлов, без пути)
    current_images = get_task_images(task_id)

    if request.method == "POST":
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        answer = request.form.get('answer', '').strip()
        difficulty = request.form.get('difficulty', '').strip()
        tags = request.form.get('tags', '').strip()
        source = request.form.get('source', '').strip()

        # Если поля обязательны
        if not title or not description:
            flash("Заполните название и описание", "error")
            conn.close()
            return render_template('edit_task.html', task=task, task_images=current_images)

        try:
            difficulty_int = int(difficulty)
            if not 1 <= difficulty_int <= 10:
                raise ValueError
        except ValueError:
            flash("Сложность от 1 до 10", "error")
            conn.close()
            return render_template('edit_task.html', task=task, task_images=current_images)

        # Обновляем текстовые поля
        cur.execute("UPDATE tasks6 SET title = ?, description = ?, anwser = ?, difficulty = ?, tags = ?, source = ? WHERE rowid = ?",
                    (title, description, answer, difficulty_int, tags, source, task_id))
        conn.commit()

        # 1) Обработка удаления изображений
        # Поддерживаем: список delete_images (много) и старую checkbox delete_image
        delete_list = request.form.getlist('delete_images') or []
        # если старый флаг стоял, удаляем файл, который хранится в task['image_path']
        if request.form.get('delete_image') == 'on' and task.get('image_path'):
            try:
                if os.path.exists(task['image_path']):
                    os.remove(task['image_path'])
            except Exception:
                pass
            # если есть колонка image_path — обнулим её
            if 'image_path' in cols:
                cur.execute("UPDATE tasks6 SET image_path = NULL WHERE rowid = ?", (task_id,))
                conn.commit()

        # удалим все файлы, имена которых указаны в delete_list (они — имена файлов, типа "89_1.jpg")
        for fname in delete_list:
            safe_name = os.path.basename(fname)  # защита
            path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        # 2) Обработка новых загруженных файлов
        uploaded = []
        uploaded += request.files.getlist('images') if 'images' in request.files else []
        single = request.files.get('image')
        if single and single.filename:
            uploaded.append(single)

        # вычисляем следующий индекс для нумерации
        remaining_images = [p for p in get_task_images(task_id)]
        # определить текущий максимум индекса
        max_index = 0
        for im in remaining_images:
            m = re.match(rf'^{re.escape(str(task_id))}_(\d+)\.(.+)$', im)
            if m:
                try:
                    v = int(m.group(1))
                    if v > max_index:
                        max_index = v
                except Exception:
                    pass
            else:
                # учёт старого файла "<id>.<ext>" — пометим как индекс 0, но при сохранении новых поставим 1+
                pass
        idx = max_index + 1 if max_index >= 1 else 1

        for f in uploaded:
            if not f or not f.filename:
                continue
            if not allowed_image(f.filename):
                # пропускаем неподдерживаемые расширения
                continue
            filename = secure_filename(f.filename)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
            dest_name = f"{task_id}_{idx}.{ext}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], dest_name)
            try:
                f.save(dest_path)
                idx += 1
            except Exception:
                continue

        # 3) После удаления/загрузки — обновляем поле image_path (если колонка есть)
        try:
            cur.execute("PRAGMA table_info('tasks6')")
            cols = [c[1] for c in cur.fetchall()]
            if 'image_path' in cols:
                imgs = get_task_images(task_id)
                if imgs:
                    first = os.path.join(app.config['UPLOAD_FOLDER'], imgs[0])
                    cur.execute("UPDATE tasks6 SET image_path = ? WHERE rowid = ?", (first, task_id))
                else:
                    cur.execute("UPDATE tasks6 SET image_path = NULL WHERE rowid = ?", (task_id,))
                conn.commit()
        except Exception:
            conn.rollback()

        conn.close()
        flash("Задание обновлено", "success")
        return redirect(url_for('edit_task', task_id=task_id))

    # GET — отображение формы с текущими изображениями (список имён)
    conn.close()
    task_images = get_task_images(task_id)
    # image_url в старом шаблоне — первый файл (если есть), иначе None
    image_url = None
    if task_images:
        image_url = os.path.join(app.config['UPLOAD_FOLDER'], task_images[0])
    return render_template('edit_task.html', task=task, task_images=task_images, image_url=image_url)



@app.route('/teacher')
def teacher_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    if not cursor.fetchone():
        conn.close()
        flash('Только учителям доступно', 'error')
        return redirect(url_for('home'))
    students = get_teacher_students(session['user_id'])
    conn.close()
    return render_template('teacher_dashboard.html', students=students, user=session.get('user_info'))

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    if delete_user(user_id):
        flash('Пользователь удален', 'success')
    else:
        flash('Ошибка удаления', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete-task/<int:task_id>', methods=['POST'])
def admin_delete_task(task_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    if delete_task(task_id):
        flash('Задание удалено', 'success')
    else:
        flash('Ошибка удаления', 'error')
    return redirect(url_for('search'))

@app.route('/admin/tasks')
def admin_tasks():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, difficulty, tags FROM tasks6")
    tasks = [{'id': row[0], 'title': row[1], 'difficulty': row[2], 'tags': row[3]} for row in cursor.fetchall()]
    conn.close()
    return render_template('admin_tasks.html', tasks=tasks, get_difficulty_color=get_difficulty_color, user=session.get('user_info'))

@app.route('/admin/delete-test/<int:test_id>', methods=['POST'])
def admin_delete_test(test_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    if delete_test(test_id):
        flash('Тест удален', 'success')
    else:
        flash('Ошибка удаления', 'error')
    return redirect(url_for('view_tests'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=9191)