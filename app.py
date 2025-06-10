from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask import send_from_directory
import sqlite3
import os
from flask import jsonify
from datetime import datetime, timedelta
import hashlib
from flask import make_response

app = Flask(__name__)
app.secret_key = 'SECRET_KEY'

# Конфигурация
app.config['UPLOAD_FOLDER'] = 'images'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_difficulty_color(difficulty):
    colors = [
        '#3498db', '#2ecc71', '#1abc9c', '#f1c40f', '#f39c12',
        '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#2c3e50'
    ]
    index = max(0, min(int(difficulty) - 1, len(colors) - 1))
    return colors[index]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

    task_title = 'В лаборатории цитологии провели эксперимент с белой планарией. Для него учёные использовали три неглубокие ёмкости, которые соединили каналами, благодаря которым планария могла свободно перемещаться из одной ёмкости в другую. В первую ёмкость налили раствор с большим количеством соли (гипертонический раствор), во вторую – с небольшим количеством соли (гипотонический раствор), а в третью ёмкость добавили столько же соли, сколько в пресном озере (изотонический раствор). Лаборант поместил планарию в первую ёмкость и стал наблюдать за её движением. Планария медленно переползла по каналу во вторую ёмкость, а затем и в третью, где осталась. Эксперимент повторили на десяти планариях и во всех случаях получили один и тот же результа'
    task_description = 'а) планарии способны к ресничному движению б) планарии невосприимчивы к соли в) планария из данного опыта обычно обитает в пресных водоёмах г) планария двигалась хаотично и случайно попала в третью ёмкость'
    task_anwser = 'в'
    task_difficulty = 4
    task_tags = 'черви'
    task_source = 'ВСОШ'
    
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
    if 'user_id' not in session or session['user_id'] is None:
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

    if test[4] and session['user_id'] is None:
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
    
    if attempts_left <= 0:
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
    if 'user_id' not in session or session['user_id'] is None:
        flash('Для просмотра результатов необходимо войти в систему', 'error')
        return redirect(url_for('login'))
    
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
    session['user_id'] = None
    session['user_info'] = {'username': 'Гость'}
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

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

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=9191)
