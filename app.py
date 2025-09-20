from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask import send_from_directory
import sqlite3
import os
from flask import jsonify
from datetime import datetime, timedelta
import hashlib
from flask import make_response
import re
import tempfile
from pathlib import Path
from flask import (
    Flask, request, render_template, redirect, url_for, flash, session, current_app
)
from flask import (
    request, session, redirect, url_for, flash, render_template,
    jsonify, current_app
)
from flask import (
    request, session, redirect, url_for, flash, render_template, current_app, abort
)
import fitz  # PyMuPDF

app = Flask(__name__)
app.secret_key = 'SECRET_KEY'

# Конфигурация
app.config['UPLOAD_FOLDER'] = 'images'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.config.setdefault("MAX_CONTENT_LENGTH", 200 * 1024 * 1024)  # 200 MB limit
ALLOWED_EXTENSIONS = {"pdf"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def table_has_column(db_path: str, table: str, column: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info('{table}')")
        cols = [r[1] for r in cur.fetchall()]
        conn.close()
        return column in cols
    except Exception:
        return False

def get_difficulty_color(difficulty):
    colors = [
        '#3498db', '#2ecc71', '#1abc9c', '#f1c40f', '#f39c12',
        '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#2c3e50'
    ]
    index = max(0, min(int(difficulty) - 1, len(colors) - 1))
    return colors[index]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def guess_ext_from_bytes(b: bytes) -> str:
    if not b or len(b) < 12:
        return "png"
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

def save_image_bytes(image_bytes: bytes, folder: str, base_name: str, ext: str = None):
    if ext is None:
        ext = guess_ext_from_bytes(image_bytes)
    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = f"{base_name}.{ext}"
    out_path = os.path.join(folder, filename)
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path

def get_test_by_id(test_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tests WHERE id = ?", (test_id,))
    test = cursor.fetchone()
    conn.close()
    return test

def get_tasks_by_ids(task_ids):
    if not task_ids:
        return []
    
    placeholders = ','.join('?' * len(task_ids))
    query = f"SELECT * FROM tasks6 WHERE id IN ({placeholders})"
    
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute(query, task_ids)
    tasks = cursor.fetchall()
    conn.close()
    return tasks

def create_attempt(test_id, user_id, score, answers):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attempts (test_id, user_id, score, answers, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (test_id, user_id, score, str(answers), datetime.now().isoformat()))
    conn.commit()
    attempt_id = cursor.lastrowid
    conn.close()
    return attempt_id

def get_user_attempts(test_id, user_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM attempts 
        WHERE test_id = ? AND user_id = ?
        ORDER BY timestamp DESC
    """, (test_id, user_id))
    attempts = cursor.fetchall()
    conn.close()
    return attempts

def get_attempt_by_id(attempt_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
    attempt = cursor.fetchone()
    conn.close()
    return attempt

def init_db():
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    
    # Таблица пользователей
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT,
            surname TEXT,
            password TEXT NOT NULL
        )
    """)
    
    # Таблица задач
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks6 (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            anwser TEXT,
            difficulty INTEGER,
            tags TEXT,
            source TEXT
        )
    """)
    
    # cursor.execute("""
    #     INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source)
    #     VALUES (?, ?, ?, ?, ?, ?)
    # """, (task_title, task_description, task_anwser, task_difficulty, task_tags, task_source))

    # cursor.execute("""DELETE FROM tasks6 WHERE id = 20""")
    
    # Таблица тестов 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time INTEGER NOT NULL,
            attempts INTEGER NOT NULL,
            tasks_id TEXT NOT NULL,
            access_code TEXT
        )
    """)
    
    # Таблица попыток прохождения тестов
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            score INTEGER NOT NULL,
            answers TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    #таблица для списка учеников
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            user_id INTEGER PRIMARY KEY,
            usi INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # cursor.execute("""
    #     INSERT INTO teachers (user_id, usi)
    #     VALUES (?, ?)
    # """, (1, 2))

    # cursor.execute("""
    #     DROP TABLE IF EXISTS teachers
    # """
    # )
    
    # Таблица связи учитель-ученик
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_students (
            teacher_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            FOREIGN KEY(teacher_id) REFERENCES users(id),
            FOREIGN KEY(student_id) REFERENCES users(id),
            PRIMARY KEY (teacher_id, student_id)
        )
    """)

    # cursor.execute("""
    #     INSERT INTO teacher_students (teacher_id, student_id)
    #     VALUES (?,?)
    # """, (1,2))
    
    conn.commit()
    conn.close()

def search_tasks(params):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    
    query = "SELECT * FROM tasks6 WHERE 1=1"
    conditions = []
    values = []
    
    if params.get('title'):
        conditions.append("title LIKE ?")
        values.append(f"%{params['title']}%")
    
    if params.get('description'):
        conditions.append("description LIKE ?")
        values.append(f"%{params['description']}%")
    
    if params.get('tag'):
        conditions.append("tags LIKE ?")
        values.append(f"%{params['tag']}%")
    
    if params.get('min_difficulty'):
        conditions.append("difficulty >= ?")
        values.append(params['min_difficulty'])
    
    if params.get('max_difficulty'):
        conditions.append("difficulty <= ?")
        values.append(params['max_difficulty'])

    # Добавим условие для поиска по источнику
    if params.get('source'):
        conditions.append("source LIKE ?")
        values.append(f"%{params['source']}%")
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    cursor.execute(query, values)
    tasks = cursor.fetchall()
    conn.close()
    return tasks

def create_test(time, attempts, tasks_id, access_code=None):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tests (time, attempts, tasks_id, access_code)
        VALUES (?, ?, ?, ?)
    """, (time, attempts, tasks_id, access_code))
    conn.commit()
    test_id = cursor.lastrowid
    conn.close()
    return test_id

def get_all_tests():
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tests")
    tests = cursor.fetchall()
    conn.close()
    return tests

def authenticate_user(username, password):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users WHERE username = ? AND password = ?", 
                  (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

def register_user(username, password, name, surname):
    try:
        conn = sqlite3.connect("datbase.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, name, surname, password) VALUES (?, ?, ?, ?)", 
                      (username, name, surname, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def process_pdf_and_create_tasks(pdf_path: str, uploader_user_id: int, db_path: str):
    """
    Открывает PDF, извлекает текст заданий и картинки, пытается сопоставить картинки с заданиями.
    Создаёт записи в БД tasks6.
    Возвращает список словарей: [{'task_id': int, 'task_num': int, 'title': str, 'desc': str, 'images': [paths]} ...]
    """
    results = []
    doc = fitz.open(pdf_path)

    # 1) Собираем весь текст по страницам — пригодится для поиска текстовых заданий
    pages_text = [doc[p].get_text("text") or "" for p in range(len(doc))]
    full_text = "\n".join(pages_text)

    # 2) Находим позиции всех явных "N." меток (начало задания). Используем многострочный режим.
    #    Регекс ловит начало строки с номером и точкой: "1.", "12." и т.д.
    pattern = re.compile(r'(?m)^\s*(\d{1,3})\.\s*')  # захватываем номер
    starts = []
    for m in pattern.finditer(full_text):
        starts.append((m.start(), int(m.group(1))))

    # Если не найдено — попробуем извлекать по страницам: найти на каждой странице "^\s*\d+\."
    if not starts:
        # fallback: искать на каждой странице отдельно
        cursor = 0
        for p_idx, txt in enumerate(pages_text):
            for m in re.finditer(r'(?m)^\s*(\d{1,3})\.\s*', txt):
                starts.append((cursor + m.start(), int(m.group(1))))
            cursor += len(txt) + 1

    # 3) Разбиваем full_text на блоки заданий по найденным позициям
    tasks_text_blocks = []
    if starts:
        for i, (pos, num) in enumerate(starts):
            start_pos = pos
            end_pos = starts[i+1][0] if i + 1 < len(starts) else len(full_text)
            block = full_text[start_pos:end_pos].strip()
            # уберём ведущую метку "N." из блока
            block = re.sub(r'^\s*\d{1,3}\.\s*', '', block, count=1, flags=re.M)
            tasks_text_blocks.append((num, block))
    else:
        # если вообще не нашли, создадим одну задачу с полным текстом
        tasks_text_blocks.append((None, full_text))

    # 4) Извлекаем изображения и пытаемся сопоставить с задачами по странице/coord (упрощённо)
    # подготовим карту: для каждой страницы — список (task_number, y0) ближайших меток (вычислим по get_text("words"))
    task_positions_by_page = {}  # page_idx -> list of (task_num, y0)
    # Создадим список глобальных задач с page_index and y0
    global_tasks = []  # (num, page_index, y0)
    # перебор страниц для поиска "N." на уровне слов (как в вашем первом скрипте)
    for p_idx in range(len(doc)):
        words = doc[p_idx].get_text("words")  # x0,y0,x1,y1,"word"
        page_candidates = []
        for w in words:
            token = w[4].strip()
            m = re.match(r'^(\d+)\.$', token)
            if m:
                num = int(m.group(1))
                y0 = w[1]
                page_candidates.append((num, y0))
                global_tasks.append((num, p_idx, y0, w[0]))  # keep x0 too
        task_positions_by_page[p_idx] = page_candidates

    saved_images = []  # list of (path, page_index, assigned_task_num or None)

    # folder to save task images
    images_root = os.path.join(app.config['UPLOAD_FOLDER'], "pdf_extracted_images")
    Path(images_root).mkdir(parents=True, exist_ok=True)

    for p_idx in range(len(doc)):
        page = doc[p_idx]
        page_dict = page.get_text("dict")
        # determine part/label omitted — используем просто номер страницы
        image_blocks = []
        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:  # image block
                bbox = block.get("bbox", [0,0,0,0])
                img_obj = block.get("image")
                image_bytes = None
                xref = None
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
                    xref = img_obj
                    try:
                        base = doc.extract_image(xref)
                        image_bytes = base.get("image")
                        ext = base.get("ext")
                    except Exception:
                        image_bytes = None
                if image_bytes and (not ext or ext == ""):
                    ext = guess_ext_from_bytes(image_bytes)
                image_blocks.append((xref, bbox, image_bytes, ext))

        # кандидаты заданий на странице
        candidates = task_positions_by_page.get(p_idx, [])
        for idx, (xref, bbox, image_bytes, ext) in enumerate(image_blocks, start=1):
            if not image_bytes:
                # try by xref if possible
                if xref:
                    try:
                        base = doc.extract_image(xref)
                        image_bytes = base.get("image")
                        ext = base.get("ext") or guess_ext_from_bytes(image_bytes)
                    except Exception:
                        image_bytes = None
            if not image_bytes:
                continue

            # координаты картинки
            img_top = bbox[1]
            img_mid = (bbox[1] + bbox[3]) / 2.0

            # найти ближайшее задание на той же странице (если есть)
            chosen_task = None
            chosen_dist = None
            if candidates:
                for (num, y0) in candidates:
                    dist = abs(img_mid - y0)
                    # если метка задания расположена ниже картинки — считаем так, чтобы избегать неверных привязок
                    if y0 > img_top:
                        dist += 1000
                    if chosen_dist is None or dist < chosen_dist:
                        chosen_dist = dist
                        chosen_task = num
            else:
                # если на странице нет меток — найдём предыдущее глобальное задание
                prev = [t for t in global_tasks if t[1] <= p_idx]
                if prev:
                    best = None
                    best_score = None
                    for (num, gp_idx, y0, x0) in prev:
                        page_diff = p_idx - gp_idx
                        score = page_diff*1000 + abs(img_mid - y0)
                        if best_score is None or score < best_score:
                            best_score = score
                            best = num
                    chosen_task = best

            base_name = f"page{p_idx+1:03d}_img{idx:02d}"
            if chosen_task:
                base_name = f"task{chosen_task:03d}_" + base_name

            # save image
            dest = save_image_bytes(image_bytes, images_root, base_name, ext)
            saved_images.append((dest, p_idx, chosen_task))

    # 5) Создаём записи в БД (tasks6). Для title возьмём "Задание N" или первую строку блока.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    created = []
    for (num, block_text) in tasks_text_blocks:
        # prepare fields
        title = f"Задание {num}" if num is not None else (block_text.splitlines()[0][:120] if block_text else "Задание")
        description = block_text
        answer = ""  # автоматически не определяем
        difficulty = 1
        tags = ""
        source = os.path.basename(pdf_path)

        cursor.execute("""
            INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, description, answer, difficulty, tags, source))
        task_id = cursor.lastrowid
        created.append({'task_id': task_id, 'task_num': num, 'title': title, 'description': description, 'images': []})

    conn.commit()
    conn.close()

    # 6) Привяжем картинки к созданным задачам по номеру task_num (если нашлись соответствия)
    # построим map номер задания -> task_id (по порядку вставки)
    num_to_task_id = {}
    # При вставке выше мы сохраняли created в том же порядке, но не гарантируем, что в created порядок соответствует номерам.
    for item in created:
        if item['task_num'] is not None:
            # найдём соответствие — возможно несколько с тем же номером, берём первое попавшееся
            num_to_task_id[item['task_num']] = item['task_id']

    # переместим/пометим найденные saved_images в папки с id задач
    for img_path, p_idx, assigned_num in saved_images:
        if assigned_num and assigned_num in num_to_task_id:
            tid = num_to_task_id[assigned_num]
            # переместим файл в папку uploads/task_images/{task_id}/
            task_dir = os.path.join(app.config['UPLOAD_FOLDER'], "task_images", str(tid))
            Path(task_dir).mkdir(parents=True, exist_ok=True)
            new_name = os.path.join(task_dir, os.path.basename(img_path))
            os.replace(img_path, new_name)
            # добавим в created
            for c in created:
                if c['task_id'] == tid:
                    c['images'].append(new_name)
        else:
            # оставим в папке pdf_extracted_images (не привязано)
            pass

    return created
           

# Добавим после других функций в app.py

# В app.py добавим
@app.route('/add-student', methods=['POST'])
def add_student():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    username = request.form['username']
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    
    if not is_teacher:
        conn.close()
        flash('Только учителя могут добавлять учеников', 'error')
        return redirect(url_for('home'))
    
    # Ищем ученика по логину
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    student = cursor.fetchone()
    
    if not student:
        conn.close()
        flash('Пользователь с таким логином не найден', 'error')
        return redirect(url_for('teacher_dashboard'))
    
    student_id = student[0]
    
    # Проверяем, не добавлен ли уже этот ученик
    cursor.execute("""
        SELECT 1 FROM teacher_students 
        WHERE teacher_id = ? AND student_id = ?
    """, (session['user_id'], student_id))
    already_added = cursor.fetchone()
    
    if already_added:
        conn.close()
        flash('Этот ученик уже добавлен', 'info')
        return redirect(url_for('teacher_dashboard'))
    
    # Добавляем связь учитель-ученик
    try:
        cursor.execute("""
            INSERT INTO teacher_students (teacher_id, student_id)
            VALUES (?, ?)
        """, (session['user_id'], student_id))
        conn.commit()
        flash('Ученик успешно добавлен', 'success')
    except sqlite3.IntegrityError:
        flash('Ошибка при добавлении ученика', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('teacher_dashboard'))

@app.route('/make-teacher/<int:user_id>')
def make_teacher(user_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO teachers (user_id) VALUES (?)", (user_id,))
        conn.commit()
        flash('Пользователь назначен учителем', 'success')
    except sqlite3.IntegrityError:
        flash('Этот пользователь уже является учителем', 'info')
    finally:
        conn.close()
    return redirect(url_for('home'))

def get_teacher_students(teacher_id):
    """Получаем список учеников учителя"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.id, users.name, users.surname, users.username
        FROM teacher_students
        JOIN users ON teacher_students.student_id = users.id
        WHERE teacher_students.teacher_id = ?
    """, (teacher_id,))
    
    # Преобразуем результаты в список словарей
    students = []
    for row in cursor.fetchall():
        students.append({
            'id': row[0],
            'name': row[1],
            'surname': row[2],
            'username': row[3]
        })
    
    conn.close()
    return students

def get_student_info(student_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, surname, username FROM users WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()
    if student:
        return {
            'id': student[0],
            'name': student[1],
            'surname': student[2],
            'username': student[3]
        }
    return None

def get_student_attempts(student_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, a.test_id, a.score, a.timestamp, t.time as duration
        FROM attempts a
        JOIN tests t ON a.test_id = t.id
        WHERE a.user_id = ?
        ORDER BY a.timestamp DESC
    """, (student_id,))
    attempts = []
    for row in cursor.fetchall():
        attempts.append({
            'id': row[0],
            'test_id': row[1],
            'score': row[2],
            'timestamp': datetime.fromisoformat(row[3]),
            'duration': row[4]
        })
    conn.close()
    return attempts

# Добавим в app.py новые функции для удаления

def delete_user(user_id):
    """Удаляет пользователя из системы"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    try:
        # Удаляем связи учитель-ученик
        cursor.execute("DELETE FROM teacher_students WHERE teacher_id = ? OR student_id = ?", 
                      (user_id, user_id))
        # Удаляем из учителей
        cursor.execute("DELETE FROM teachers WHERE user_id = ?", (user_id,))
        # Удаляем попытки тестов
        cursor.execute("DELETE FROM attempts WHERE user_id = ?", (user_id,))
        # Удаляем самого пользователя
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error deleting user: {e}")
        return False
    finally:
        conn.close()

def delete_task(task_id):
    """Удаляет задание из системы"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    try:
        # Удаляем изображение задания, если оно есть
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Удаляем задание
        cursor.execute("DELETE FROM tasks6 WHERE id = ?", (task_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error deleting task: {e}")
        return False
    finally:
        conn.close()

def delete_test(test_id):
    """Удаляет тест из системы"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    try:
        # Удаляем попытки прохождения теста
        cursor.execute("DELETE FROM attempts WHERE test_id = ?", (test_id,))
        # Удаляем сам тест
        cursor.execute("DELETE FROM tests WHERE id = ?", (test_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error deleting test: {e}")
        return False
    finally:
        conn.close()

# Добавим новые маршруты для удаления
@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    if delete_user(user_id):
        flash('Пользователь успешно удален', 'success')
    else:
        flash('Ошибка при удалении пользователя', 'error')
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete-task/<int:task_id>', methods=['POST'])
def admin_delete_task(task_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    if delete_task(task_id):
        flash('Задание успешно удалено', 'success')
    else:
        flash('Ошибка при удалении задания', 'error')
    
    return redirect(url_for('search'))

@app.route('/admin/tasks')
def admin_tasks():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, difficulty, tags FROM tasks6")
    tasks = [{
        'id': row[0],
        'title': row[1],
        'difficulty': row[2],
        'tags': row[3]
    } for row in cursor.fetchall()]
    conn.close()
    
    return render_template('admin_tasks.html',
                         tasks=tasks,
                         get_difficulty_color=get_difficulty_color,
                         user=session.get('user_info'))

@app.route('/admin/delete-test/<int:test_id>', methods=['POST'])
def admin_delete_test(test_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    if delete_test(test_id):
        flash('Тест успешно удален', 'success')
    else:
        flash('Ошибка при удалении теста', 'error')
    
    return redirect(url_for('view_tests'))

def get_student_stats(student_id):
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    
    # Общее количество тестов
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE user_id = ?", (student_id,))
    total_tests = cursor.fetchone()[0] or 0
    
    # Средний балл
    cursor.execute("SELECT AVG(score) FROM attempts WHERE user_id = ?", (student_id,))
    average_score = round(cursor.fetchone()[0] or 0, 1)
    
    # Лучший результат
    cursor.execute("SELECT MAX(score) FROM attempts WHERE user_id = ?", (student_id,))
    best_score = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_tests': total_tests,
        'average_score': average_score,
        'best_score': best_score
    }

def get_student_results(student_id):
    """Получаем результаты тестов ученика"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT attempts.id, attempts.test_id, attempts.score, attempts.timestamp, tests.time
        FROM attempts
        JOIN tests ON attempts.test_id = tests.id
        WHERE attempts.user_id = ?
        ORDER BY attempts.timestamp DESC
    """, (student_id,))
    attempts = cursor.fetchall()
    conn.close()
    return attempts

# Новые маршруты
@app.route('/teacher')
def teacher_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    
    if not is_teacher:
        conn.close()
        flash('Эта страница доступна только учителям', 'error')
        return redirect(url_for('home'))
    
    # Получаем список учеников
    students = get_teacher_students(session['user_id'])
    conn.close()
    
    return render_template('teacher_dashboard.html', 
                         students=students,
                         user=session.get('user_info'))

# Добавим в app.py новые функции и маршруты

def is_admin():
    """Проверяет, является ли текущий пользователь админом"""
    return 'user_info' in session and session['user_info'].get('username') == 'admin'

def is_guest():
    """Проверяет, является ли текущий пользователь админом"""
    return 'user_info' in session and session['user_info'].get('username') == 'admin'

def get_all_users():
    """Получает список всех пользователей"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users")
    users = [{
        'id': row[0],
        'username': row[1],
        'name': row[2],
        'surname': row[3],
        'is_teacher': is_user_teacher(row[0])
    } for row in cursor.fetchall()]
    conn.close()
    return users

def is_user_teacher(user_id):
    """Проверяет, является ли пользователь учителем"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (user_id,))
    result = cursor.fetchone() is not None
    conn.close()
    return result

def toggle_teacher_role(user_id):
    """Переключает роль учителя у пользователя"""
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    
    if is_user_teacher(user_id):
        cursor.execute("DELETE FROM teachers WHERE user_id = ?", (user_id,))
        action = "removed"
    else:
        try:
            cursor.execute("INSERT INTO teachers (user_id) VALUES (?)", (user_id,))
            action = "added"
        except sqlite3.IntegrityError:
            action = "exists"
    
    conn.commit()
    conn.close()
    return action

# Новые маршруты
@app.route('/admin')
def admin_panel():
    if not is_admin():
        flash('Доступ запрещен. Только администратор может просматривать эту страницу.', 'error')
        return redirect(url_for('home'))
    
    users = get_all_users()
    return render_template('admin_panel.html', 
                         users=users,
                         user=session.get('user_info'))

@app.route('/admin/toggle-teacher/<int:user_id>')
def admin_toggle_teacher(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    action = toggle_teacher_role(user_id)
    if action == "added":
        flash('Пользователь назначен учителем', 'success')
    elif action == "removed":
        flash('У пользователя удалена роль учителя', 'success')
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/view-user/<int:user_id>')
def admin_view_user(user_id):
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    # Получаем информацию о пользователе
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, surname FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('admin_panel'))
    
    user_info = {
        'id': user[0],
        'username': user[1],
        'name': user[2],
        'surname': user[3],
        'is_teacher': is_user_teacher(user_id)
    }
    
    # Получаем список учеников (если это учитель)
    students = []
    if user_info['is_teacher']:
        cursor.execute("""
            SELECT u.id, u.username, u.name, u.surname
            FROM teacher_students ts
            JOIN users u ON ts.student_id = u.id
            WHERE ts.teacher_id = ?
        """, (user_id,))
        students = [{
            'id': row[0],
            'username': row[1],
            'name': row[2],
            'surname': row[3]
        } for row in cursor.fetchall()]
    
    # Получаем результаты тестов
    cursor.execute("""
        SELECT a.id, a.test_id, a.score, a.timestamp, t.time
        FROM attempts a
        JOIN tests t ON a.test_id = t.id
        WHERE a.user_id = ?
        ORDER BY a.timestamp DESC
    """, (user_id,))
    attempts = [{
        'id': row[0],
        'test_id': row[1],
        'score': row[2],
        'timestamp': row[3],
        'time': row[4]
    } for row in cursor.fetchall()]
    
    conn.close()
    
    return render_template('admin_user_view.html',
                         user_info=user_info,
                         students=students,
                         attempts=attempts,
                         current_user=session.get('user_info'))

@app.route('/admin/add-student', methods=['POST'])
def admin_add_student():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    teacher_id = request.form.get('teacher_id')
    student_username = request.form.get('student_username')
    
    # Проверяем, что учитель существует
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (teacher_id,))
    if not cursor.fetchone():
        flash('Указанный пользователь не является учителем', 'error')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    
    # Находим ученика по логину
    cursor.execute("SELECT id FROM users WHERE username = ?", (student_username,))
    student = cursor.fetchone()
    if not student:
        flash('Ученик с таким логином не найден', 'error')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    
    student_id = student[0]
    
    # Проверяем, не добавлен ли уже этот ученик
    cursor.execute("""
        SELECT 1 FROM teacher_students 
        WHERE teacher_id = ? AND student_id = ?
    """, (teacher_id, student_id))
    if cursor.fetchone():
        flash('Этот ученик уже добавлен', 'info')
        return redirect(url_for('admin_view_user', user_id=teacher_id))
    
    # Добавляем связь
    try:
        cursor.execute("""
            INSERT INTO teacher_students (teacher_id, student_id)
            VALUES (?, ?)
        """, (teacher_id, student_id))
        conn.commit()
        flash('Ученик успешно добавлен', 'success')
    except sqlite3.Error as e:
        flash(f'Ошибка при добавлении ученика: {str(e)}', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('admin_view_user', user_id=teacher_id))

@app.route('/admin/remove-student', methods=['POST'])
def admin_remove_student():
    if not is_admin():
        flash('Доступ запрещен', 'error')
        return redirect(url_for('home'))
    
    teacher_id = request.form.get('teacher_id')
    student_id = request.form.get('student_id')
    
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM teacher_students
            WHERE teacher_id = ? AND student_id = ?
        """, (teacher_id, student_id))
        conn.commit()
        flash('Ученик успешно удален', 'success')
    except sqlite3.Error as e:
        flash(f'Ошибка при удалении ученика: {str(e)}', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('admin_view_user', user_id=teacher_id))

@app.route('/teacher/student/<int:student_id>')
def view_student_results(student_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    
    # Проверяем, что ученик привязан к учителю
    cursor.execute("""
        SELECT 1 FROM teacher_students 
        WHERE teacher_id = ? AND student_id = ?
    """, (session['user_id'], student_id))
    has_access = cursor.fetchone() is not None
    conn.close()
    
    if not is_teacher or not has_access:
        flash('У вас нет доступа к этим результатам', 'error')
        return redirect(url_for('home'))
    
    student = get_student_info(student_id)
    if not student:
        flash('Ученик не найден', 'error')
        return redirect(url_for('teacher_dashboard'))
    
    attempts = get_student_attempts(student_id)
    stats = get_student_stats(student_id)
    
    return render_template('student_results.html', 
                         student=student,
                         attempts=attempts,
                         stats=stats)

@app.route('/get-test-html/<int:test_id>')
def get_test_html(test_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    
    if not is_teacher:
        conn.close()
        return "Доступ запрещен", 403
    
    # Получаем данные теста
    test = get_test_by_id(test_id)
    if not test:
        return "Тест не найден", 404
    
    # Получаем задания для теста
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    
    # Рендерим специальный шаблон для PDF
    return render_template('test_pdf.html', 
                         test=test,
                         tasks=tasks,
                         get_difficulty_color=get_difficulty_color)

@app.route('/remove-student/<int:student_id>', methods=['POST'])
def remove_student(student_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    
    if not is_teacher:
        conn.close()
        flash('Только учителя могут удалять учеников', 'error')
        return redirect(url_for('home'))
    
    # Удаляем связь учитель-ученик
    cursor.execute("""
        DELETE FROM teacher_students 
        WHERE teacher_id = ? AND student_id = ?
    """, (session['user_id'], student_id))
    conn.commit()
    conn.close()
    
    flash('Ученик успешно удален', 'success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/create-task', methods=['GET', 'POST'])
def create_task():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teachers WHERE user_id = ?", (session['user_id'],))
    teacher = cursor.fetchone()
    conn.close()
    
    if not teacher:
        flash('Только учителя могут добавлять задания', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        answer = request.form['answer']
        difficulty = request.form['difficulty']
        tags = request.form['tags']
        source = request.form['source']
        image = request.files.get('image')
        
        # Валидация данных
        if not all([title, description, answer, difficulty]):
            flash('Пожалуйста, заполните все обязательные поля', 'error')
            return render_template('create_task.html')
        
        try:
            difficulty = int(difficulty)
            if difficulty < 1 or difficulty > 10:
                flash('Сложность должна быть от 1 до 10', 'error')
                return render_template('create_task.html')
        except ValueError:
            flash('Сложность должна быть числом', 'error')
            return render_template('create_task.html')
        
        # Сохраняем задание в базу данных
        conn = sqlite3.connect("datbase.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tasks6 (title, description, anwser, difficulty, tags, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, description, answer, difficulty, tags, source))
        
        # Получаем ID нового задания
        task_id = cursor.lastrowid
        
        # Сохраняем изображение, если оно есть
        if image and allowed_file(image.filename):
            filename = f"{task_id}.{secure_filename(image.filename).rsplit('.', 1)[1].lower()}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            # flash(f'Изображение сохранено как {filename}', 'success')
        
        conn.commit()
        conn.close()
        
        # flash('Задание успешно добавлено!', 'success')
        return redirect(url_for('home'))
    
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
    
    # Получаем время начала теста из сессии
    start_time = session.get(f'test_{test_id}_start_time')
    if not start_time:
        return jsonify({'error': 'Test not started'}), 400
    
    elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
    time_left = max(0, test[1] * 60 - elapsed)  # test[1] - время в минутах
    
    return jsonify({
        'time_left': time_left,
        'test_duration': test[1] * 60
    })

# Добавляем после других маршрутов в app.py

@app.route('/profile')
def profile():
    if 'user_id' not in session or session['user_id'] == 0:
        flash('Для доступа к профилю необходимо войти в систему', 'error')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Получаем все попытки пользователя
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT attempts.id, attempts.test_id, attempts.score, attempts.timestamp, tests.time, tests.attempts
        FROM attempts
        JOIN tests ON attempts.test_id = tests.id
        WHERE attempts.user_id = ?
        ORDER BY attempts.timestamp DESC
    """, (user_id,))
    attempts = cursor.fetchall()
    conn.close()
    
    return render_template('profile.html', 
                         attempts=attempts, 
                         user=session.get('user_info'))
    
@app.route('/test/<int:test_id>', methods=['GET', 'POST'])
def take_test(test_id):
    if 'user_id' not in session:
        flash('Для прохождения теста необходимо войти в систему', 'error')
        return redirect(url_for('login'))

    test = get_test_by_id(test_id)
    if not test:
        flash('Тест не найден', 'error')
        return redirect(url_for('view_tests'))
    
    # if test[4] and is_guest():
    #     return redirect(url_for('login'))

    if test[4] and session['user_id'] == 0:
        return redirect(url_for('login'))

    # Проверка доступа для приватных тестов
    if test[4] and not session.get(f'test_{test_id}_access'):
        return redirect(url_for('start_test', test_id=test_id))
    
    # Получаем список ID заданий
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    
    if not tasks:
        flash('Задания для теста не найдены', 'error')
        return redirect(url_for('view_tests'))
    
    # Проверяем попытки
    user_attempts = get_user_attempts(test_id, session['user_id'])
    attempts_left = test[2] - len(user_attempts)
    
    if attempts_left <= 0 and session['user_id'] != 0:
        flash('У вас больше нет попыток для прохождения этого теста', 'error')
        return redirect(url_for('view_tests'))
    
    if request.method == 'POST':
        user_answers = {}
        correct_count = 0
        
        # Проверяем ответы
        for task in tasks:
            answer_key = f"answer_{task[0]}"
            user_answer = request.form.get(answer_key, '').strip()
            user_answers[task[0]] = user_answer
            
            # Сравниваем с правильным ответом (игнорируем регистр и пробелы)
            if user_answer.lower() == (task[3] or '').lower():
                correct_count += 1
        
        # Рассчитываем результат в процентах
        score = int((correct_count / len(tasks)) * 100) if tasks else 0

        
        # Сохраняем попытку
        attempt_id = create_attempt(test_id, session['user_id'], score, user_answers)
        
        # Перенаправляем на страницу результатов
        return redirect(url_for('test_result', attempt_id=attempt_id))
    
    if request.method == 'GET':
        # Сохраняем время начала теста
        session[f'test_{test_id}_start_time'] = datetime.now().isoformat()
    
    # Для GET запроса показываем тест
    return render_template('take_test.html', 
                         test=test, 
                         tasks=tasks, 
                         get_difficulty_color=get_difficulty_color,
                         attempts_left=attempts_left,
                         user=session.get('user_info'))

@app.route('/test/result/<int:attempt_id>')
def test_result(attempt_id):
    # if 'user_id' not in session or session['user_id'] == 0:
    #     flash('Для просмотра результатов необходимо войти в систему', 'error')
    #     return redirect(url_for('login'))
    
    attempt = get_attempt_by_id(attempt_id)
    if not attempt:
        flash('Результат теста не найден', 'error')
        return redirect(url_for('view_tests'))
    
    # Проверяем, что результат принадлежит текущему пользователю
    if attempt[2] != session['user_id']:
        flash('У вас нет доступа к этим результатам', 'error')
        return redirect(url_for('view_tests'))
    
    test = get_test_by_id(attempt[1])
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    
    # Парсим ответы пользователя
    import ast
    user_answers = ast.literal_eval(attempt[4])
    
    return render_template('test_result.html', 
                         test=test,
                         attempt=attempt,
                         tasks=tasks,
                         user_answers=user_answers,
                         get_difficulty_color=get_difficulty_color,
                         user=session.get('user_info'))

    
@app.route('/create-test', methods=['GET', 'POST'])
def create_test_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teachers WHERE user_id = ?", (session['user_id'],))
    teacher = cursor.fetchone()
    conn.close()
    
    if not teacher:
        flash('Только учителя могут добавлять задания', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        time = request.form.get('time')
        attempts = request.form.get('attempts')
        tasks_id = request.form.get('tasks_id')
        access_code = request.form.get('access_code', None)
        
        if not time or not attempts or not tasks_id:
            flash('Пожалуйста, заполните все обязательные поля', 'error')
        else:
            try:
                create_test(int(time), int(attempts), tasks_id, access_code)
                # flash('Тест успешно создан!', 'success')
                return redirect(url_for('home'))
            except ValueError:
                flash('Пожалуйста, введите корректные числовые значения', 'error')
    
    return render_template('create_test.html', user=session.get('user_info'))


@app.route('/start-test/<int:test_id>', methods=['GET', 'POST'])
def start_test(test_id):
    if 'user_id' not in session:
        flash('Для прохождения теста необходимо войти в систему', 'error')
        return redirect(url_for('login'))

    test = get_test_by_id(test_id)
    if not test:
        flash('Тест не найден', 'error')
        return redirect(url_for('view_tests'))

    # Если тест приватный и код не введен/неверный
    if test[4]:  # test[4] - access_code
        if request.method == 'POST':
            entered_code = request.form.get('access_code', '').strip()
            if entered_code == test[4]:
                session[f'test_{test_id}_access'] = True
                return redirect(url_for('take_test', test_id=test_id))
            # else:
            #     flash('Неверный код доступа', 'error')
        return render_template('access_code.html', test=test)

    # Если тест публичный
    return redirect(url_for('take_test', test_id=test_id))

@app.route('/submit-test/<int:test_id>', methods=['POST'])
def submit_test(test_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Получаем ответы пользователя
    user_answers = {}
    for key, value in request.form.items():
        if key.startswith('answer_'):
            task_id = int(key.replace('answer_', ''))
            user_answers[task_id] = value
    
    # Получаем правильные ответы из базы данных
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    
    # Получаем задания теста
    cursor.execute("SELECT tasks_id FROM tests WHERE id = ?", (test_id,))
    test = cursor.fetchone()
    task_ids = test[0].split(',')
    
    # Получаем правильные ответы
    placeholders = ','.join(['?'] * len(task_ids))
    query = f"SELECT id, anwser FROM tasks6 WHERE id IN ({placeholders})"
    cursor.execute(query, task_ids)
    correct_answers = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Сравниваем ответы
    results = []
    total = len(correct_answers)
    correct = 0
    
    for task_id, correct_answer in correct_answers.items():
        user_answer = user_answers.get(task_id, '')
        is_correct = (user_answer.lower() == correct_answer.lower())
        results.append({
            'task_id': task_id,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct
        })
        if is_correct:
            correct += 1
    
    score = round((correct / total) * 100) if total > 0 else 0
    
    return render_template('test_results.html',
                         test_id=test_id,
                         results=results,
                         total=total,
                         correct=correct,
                         score=score,
                         user=session.get('user_info'))

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
    
    # Проверяем, является ли пользователь учителем
    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None
    conn.close()
    
    return render_template('index.html', 
                         user=session.get('user_info'),
                         is_teacher=is_teacher)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        
        if user:
            session['user_id'] = user[0]
            session['user_info'] = {
                'username': user[1],
                'name': user[2],
                'surname': user[3]
            }
            return redirect(url_for('home'))
        else:
            flash('Неверное имя пользователя или пароль', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form.get('name', '')
        surname = request.form.get('surname', '')
        
        if register_user(username, password, name, surname):
            flash('Регистрация успешна! Теперь вы можете войти.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Имя пользователя уже занято', 'error')
    
    return render_template('register.html')

@app.route('/guest')
def guest():
    session['user_id'] = 0
    # session['user_id'] = None
    session['user_info'] = {'username': 'Гость'}
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/upload-pdf", methods=["GET", "POST"])
def upload_pdf():
    """
    Диагностический / устойчивый маршрут загрузки PDF.
    Возвращает JSON при AJAX (X-Requested-With) с подробной диагностикой.
    """
    # проверка авторизации
    if 'user_id' not in session:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'unauthenticated', 'message': 'login required'}), 401
        return redirect(url_for('login'))

    # проверка учителя
    try:
        conn = sqlite3.connect("datbase.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teachers WHERE user_id = ?", (session['user_id'],))
        teacher = cursor.fetchone()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not teacher:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'forbidden', 'message': 'Только учителя'}), 403
        flash("Только учителя могут загружать PDF для авторазбора", "error")
        return redirect(url_for('home'))

    if request.method == "POST":
        # Попытаемся достать файл из нескольких возможных ключей (на случай, если front-end отправил другое имя)
        file = None
        tried_keys = []
        for key in ("file", "file0", "file1", "file[]", "upload", "pdf"):
            tried_keys.append(key)
            if key in request.files:
                file = request.files.get(key)
                break
        # если не нашли — попробуем взять первый файл в request.files
        if not file and request.files:
            tried_keys.append("first_in_request_files")
            file = next(iter(request.files.values()))

        # Диагностика: какие ключи пришли
        received_keys = list(request.files.keys())

        if not file:
            msg = "Файл не обнаружен в request.files. Ключи: " + str(received_keys)
            current_app.logger.debug(msg)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': msg, 'received_keys': received_keys}), 400
            flash(msg, "error")
            return redirect(request.url)

        # Соберём диагностическую информацию
        filename = file.filename or ""
        mimetype = file.mimetype or ""
        diagnostics = {
            "received_keys": received_keys,
            "used_key": key if 'key' in locals() else 'unknown',
            "filename_raw": filename,
            "mimetype": mimetype,
            "allowed_by_extension": allowed_file(filename),
        }

        # Если имя файла пустое или расширение не pdf — попробуем детектировать по первым байтам
        first_bytes = None
        try:
            # прочитаем первые 16 байт
            stream = file.stream
            stream.seek(0)
            first_bytes = stream.read(16)
            # вернём поток в начало, чтобы потом можно было сохранить
            try:
                stream.seek(0)
            except Exception:
                pass
            diagnostics['first_bytes_prefix'] = first_bytes[:8].hex() if isinstance(first_bytes, (bytes, bytearray)) else str(first_bytes)
            diagnostics['looks_like_pdf_by_magic'] = isinstance(first_bytes, (bytes, bytearray)) and first_bytes.startswith(b'%PDF')
        except Exception as e:
            diagnostics['read_first_bytes_error'] = str(e)
            current_app.logger.exception("Не удалось прочитать первые байты файла")

        # Если имя пустое или расширение не pdf, но magic bytes показывают PDF — позволим
        is_pdf_magic = diagnostics.get('looks_like_pdf_by_magic', False)
        if not diagnostics['allowed_by_extension'] and not is_pdf_magic:
            # не pdf по расширению и не pdf по magic
            msg = f"Только PDF разрешены. filename='{filename}', mimetype='{mimetype}', allowed_by_extension={diagnostics['allowed_by_extension']}, magic_pdf={is_pdf_magic}"
            current_app.logger.debug("Upload rejected: " + msg)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                diagnostics['message'] = msg
                return jsonify({'status': 'error', 'message': msg, 'diagnostics': diagnostics}), 400
            flash(msg, "error")
            return redirect(request.url)

        # Сохраняем временно файл и вызываем обработчик
        tmp_dir = tempfile.mkdtemp(prefix="pdf_upload_")
        pdf_filename = secure_filename(filename) or "uploaded.pdf"
        pdf_path = os.path.join(tmp_dir, pdf_filename)
        try:
            file.save(pdf_path)
            current_app.logger.debug(f"Saved uploaded file to {pdf_path}")
            # вызов вашей функции обработки (предполагаем, что она есть)
            created = process_pdf_and_create_tasks(pdf_path, session['user_id'], db_path="datbase.db")

            # возвращаем JSON при AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                created_summary = [{'task_id': c.get('task_id'), 'task_num': c.get('task_num')} for c in created]
                return jsonify({
                    'status': 'ok',
                    'created': created_summary,
                    'diagnostics': diagnostics
                }), 200

            # обычный POST -> redirect + flash
            flash(f"Обработано: создано {len(created)} заданий.", "success")
            flash(f"IDs созданных заданий: {[c.get('task_id') for c in created]}", "success")
            return redirect(url_for('home'))

        except Exception as e:
            current_app.logger.exception("Ошибка при обработке PDF")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                diagnostics['exception'] = str(e)
                return jsonify({'status': 'error', 'message': str(e), 'diagnostics': diagnostics}), 500
            flash(f"Ошибка при обработке PDF: {e}", "error")
            return redirect(request.url)
        finally:
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
            except Exception:
                current_app.logger.exception("Ошибка при очистке временных файлов")

    # GET
    return render_template("upload_pdf.html")

@app.route('/print-test/<int:test_id>')
def print_test(test_id):
    # Только авторизованные учителя могут печатать
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect("datbase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE user_id = ?", (session['user_id'],))
    is_teacher = cursor.fetchone() is not None

    if not is_teacher:
        conn.close()
        return "Доступ запрещен", 403

    test = get_test_by_id(test_id)
    if not test:
        conn.close()
        return "Тест не найден", 404

    # Получаем задания теста
    task_ids = [int(id_str.strip()) for id_str in test[3].split(',') if id_str.strip()]
    tasks = get_tasks_by_ids(task_ids)
    conn.close()

    # Возвращаем минималистичный шаблон для печати
    return render_template('print_test.html',
                           test=test,
                           tasks=tasks,
                           user=session.get('user_info'))


@app.route('/search', methods=['GET'])
def search():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    search_params = {
        'title': request.args.get('title', ''),
        'description': request.args.get('description', ''),
        'tag': request.args.get('tag', ''),
        'source': request.args.get('source', ''),  # Добавили параметр источника
        'min_difficulty': request.args.get('min_difficulty', ''),
        'max_difficulty': request.args.get('max_difficulty', '')
    }
    
    tasks = search_tasks(search_params)
    return render_template('tasks.html', 
                         tasks=tasks, 
                         get_difficulty_color=get_difficulty_color,
                         search_params=search_params,
                         user=session.get('user_info'))

@app.route('/edit-task/<int:task_id>', methods=['GET', 'POST'])
def edit_task(task_id):
    # проверка авторизации
    if 'user_id' not in session:
        flash("Требуется авторизация", "error")
        return redirect(url_for('login'))

    # проверка, учитель ли
    try:
        conn = sqlite3.connect("datbase.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM teachers WHERE user_id = ?", (session['user_id'],))
        teacher = cur.fetchone()
        conn.close()
    except Exception:
        current_app.logger.exception("DB error while checking teacher")
        flash("Ошибка сервера", "error")
        return redirect(url_for('home'))

    if not teacher:
        flash("Только учителя могут редактировать задания", "error")
        return redirect(url_for('home'))

    # Получаем задачу
    try:
        conn = sqlite3.connect("datbase.db")
        cur = conn.cursor()
        cur.execute("SELECT rowid, * FROM tasks6 WHERE rowid = ?", (task_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            abort(404)
        # Получим имена колонок, чтобы корректно сопоставить
        cur.execute("PRAGMA table_info('tasks6')")
        cols_info = cur.fetchall()  # (cid, name, type, ...)
        cols = [c[1] for c in cols_info]
        # Создадим словарь task из row (row[0] == rowid)
        # row structure: (rowid, col1, col2, ...)
        task = {}
        # row[0] is rowid; subsequent indexes correspond to cols order
        for idx, colname in enumerate(cols, start=1):
            task[colname] = row[idx]
        task['id'] = task_id
        conn.close()
    except Exception:
        current_app.logger.exception("DB error while fetching task")
        flash("Ошибка сервера при получении задания", "error")
        return redirect(url_for('home'))

    # POST — обновляем
    if request.method == "POST":
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        answer = request.form.get('answer', '').strip()
        difficulty = request.form.get('difficulty', '').strip()
        tags = request.form.get('tags', '').strip()
        source = request.form.get('source', '').strip()
        delete_image = request.form.get('delete_image') == 'on'
        image_file = request.files.get('image')

        # Валидация
        if not title or not description:
            flash("Поля 'Условие' и 'Варианты ответа' обязательны", "error")
            return render_template('edit_task.html', task=task)

        try:
            difficulty_int = int(difficulty)
            if difficulty_int < 1 or difficulty_int > 10:
                raise ValueError()
        except Exception:
            flash("Сложность должна быть числом от 1 до 10", "error")
            return render_template('edit_task.html', task=task)

        # Сохраняем изменения в БД
        try:
            conn = sqlite3.connect("datbase.db")
            cur = conn.cursor()
            # Обновим стандартные поля. В вашей БД ответ хранится в колонке 'anwser' (обратите внимание на опечатку).
            # Если в вашей схеме другое имя, поправьте ниже.
            update_sql = """
                UPDATE tasks6
                SET title = ?, description = ?, anwser = ?, difficulty = ?, tags = ?, source = ?
                WHERE rowid = ?
            """
            cur.execute(update_sql, (title, description, answer, difficulty_int, tags, source, task_id))
            conn.commit()
        except Exception:
            current_app.logger.exception("DB error while updating task")
            flash("Ошибка при сохранении задания", "error")
            try:
                conn.close()
            except Exception:
                pass
            return render_template('edit_task.html', task=task)
        
        # Обработка изображения (сохранение / удаление)
        # Папка для изображений по заданию
        img_parent = os.path.join(app.config.get('UPLOAD_FOLDER', 'uploads'), "task_images", str(task_id))
        ensure_dir(img_parent)

        # Определим, есть ли колонка image_path в таблице
        has_image_path_col = table_has_column("datbase.db", "tasks6", "image_path")

        # Удаление изображения, если выбран checkbox
        if delete_image:
            # если в task есть image_path и файл существует — удалим
            existing_image_path = task.get('image_path')
            if existing_image_path and os.path.exists(existing_image_path):
                try:
                    os.remove(existing_image_path)
                except Exception:
                    current_app.logger.exception("Не удалось удалить старое изображение")
            # очистим поле image_path в БД, если есть колонка
            if has_image_path_col:
                try:
                    cur.execute("UPDATE tasks6 SET image_path = NULL WHERE rowid = ?", (task_id,))
                    conn.commit()
                except Exception:
                    current_app.logger.exception("Не удалось очистить image_path")
        
        # Сохранение новой загрузки
        if image_file and image_file.filename:
            if not allowed_image(image_file.filename):
                flash("Неверный формат изображения. Разрешены: JPG, PNG, JPEG", "error")
                # закрываем и возвращаем
                try:
                    conn.close()
                except Exception:
                    pass
                return render_template('edit_task.html', task=task)
            filename = secure_filename(image_file.filename)
            dest = os.path.join(img_parent, filename)
            try:
                image_file.save(dest)
                # Попробуем сохранить путь в БД (если есть колонка)
                if has_image_path_col:
                    try:
                        cur.execute("UPDATE tasks6 SET image_path = ? WHERE rowid = ?", (dest, task_id))
                        conn.commit()
                    except Exception:
                        current_app.logger.exception("Не удалось записать image_path в БД")
                flash("Задание и изображение успешно обновлены", "success")
            except Exception:
                current_app.logger.exception("Ошибка при сохранении изображения")
                flash("Ошибка при сохранении изображения", "error")
        else:
            flash("Задание успешно обновлено", "success")

        try:
            conn.close()
        except Exception:
            pass

        return redirect(url_for('edit_task', task_id=task_id))

    # GET — отрисовать форму с текущими значениями
    # определим путь к изображению, если он есть (и колонка image_path присутствует)
    image_url = None
    if task.get('image_path') and os.path.exists(task.get('image_path')):
        # преобразование для отображения: относительный путь от /uploads/... предполагается статически доступен
        image_url = '/' + os.path.relpath(task.get('image_path'), start=os.getcwd()).replace(os.path.sep, '/')

    return render_template('edit_task.html', task=task, image_url=image_url)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=9191)
