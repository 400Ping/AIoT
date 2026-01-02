#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    redirect,
    url_for,
    abort,
)
from pathlib import Path
import sqlite3
import csv
from datetime import datetime, date
from line_notify import send_line_message
import os
from dotenv import load_dotenv, find_dotenv

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "violations.db"
CSV_PATH = DATA_DIR / "violations.csv"

load_dotenv(find_dotenv())

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")  # demo 用

# ---------- Flask-Login 設定 ----------
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


class User(UserMixin):
    def __init__(self, id, username, password_hash, is_admin: int):
        self.id = str(id)
        self.username = username
        self.password_hash = password_hash
        self.is_admin = bool(is_admin)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_user_by_id(user_id: str) -> User | None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, password_hash, is_admin FROM users WHERE id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return User(row["id"], row["username"], row["password_hash"], row["is_admin"])
    return None


def get_user_by_username(username: str) -> User | None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return User(row["id"], row["username"], row["password_hash"], row["is_admin"])
    return None


@login_manager.user_loader
def load_user(user_id: str):
    return get_user_by_id(user_id)


# ---------- DB 初始 & CSV 重寫 ----------


def init_storage():
    DATA_DIR.mkdir(exist_ok=True)

    conn = get_db_connection()
    cur = conn.cursor()

    # 事件表
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera_id TEXT,
            status TEXT,
            num_people INTEGER,
            num_no_helmet INTEGER,
            image_url TEXT
        )
        """
    )

    # 使用者表（會員系統）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    conn.commit()

    # 如果還沒有任何使用者，就自動建立一個 admin / admin123
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    if count == 0:
        default_username = "admin"
        default_password = "admin123"
        pw_hash = generate_password_hash(default_password)
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            (default_username, pw_hash),
        )
        conn.commit()
        print(
            f"[INFO] Created default admin user: "
            f"{default_username} / {default_password}"
        )

    conn.close()

    # 沒有 CSV 就建一個空的（有 header）
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "timestamp",
                    "camera_id",
                    "status",
                    "num_people",
                    "num_no_helmet",
                    "image_url",
                ]
            )


def rewrite_csv_from_db():
    """每次資料有變動（新增 / 刪除 / 清空）後，重寫 CSV，保持一致。"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, timestamp, camera_id, status,
               num_people, num_no_helmet, image_url
        FROM events
        ORDER BY datetime(timestamp) DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "timestamp",
                "camera_id",
                "status",
                "num_people",
                "num_no_helmet",
                "image_url",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                ]
            )


def fetch_events():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, timestamp, camera_id, status,
               num_people, num_no_helmet, image_url
        FROM events
        ORDER BY datetime(timestamp) DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    events = [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "camera_id": r["camera_id"],
            "status": r["status"],
            "num_people": r["num_people"],
            "num_no_helmet": r["num_no_helmet"],
            "image_url": r["image_url"],
        }
        for r in rows
    ]
    return events


def insert_event(event: dict) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (timestamp, camera_id, status,
                            num_people, num_no_helmet, image_url)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event["timestamp"],
            event.get("camera_id"),
            event.get("status"),
            event.get("num_people", 0),
            event.get("num_no_helmet", 0),
            event.get("image_url"),
        ),
    )
    event_id = cur.lastrowid
    conn.commit()
    conn.close()

    # 每次插入後重寫 CSV
    rewrite_csv_from_db()
    return event_id


# ---------- 一般頁面 ----------


@app.route("/")
def index():
    events = fetch_events()
    today_str = date.today().isoformat()
    today_count = sum(
        1
        for e in events
        if e["status"] == "unsafe" and e["timestamp"].startswith(today_str)
    )
    last_event = events[0] if events else None
    return render_template(
        "dashboard.html",
        today_count=today_count,
        last_event=last_event,
    )


@app.route("/events")
def events_page():
    events = fetch_events()
    return render_template("events.html", events=events)


@app.route("/stats")
def stats_page():
    conn = get_db_connection()
    cur = conn.cursor()

    # (1) 今天的違規統計（按小時）
    today_str = date.today().isoformat()
    cur.execute(
        """
        SELECT substr(timestamp, 1, 13) as hour, COUNT(*)
        FROM events
        WHERE status = 'unsafe' AND timestamp LIKE ?
        GROUP BY hour
        ORDER BY hour
        """,
        (today_str + "%",),
    )
    rows_today = cur.fetchall()
    labels_today = [r[0][-2:] + ":00" for r in rows_today]  # HH:00
    values_today = [r[1] for r in rows_today]

    # (2) 歷史所有違規統計（按日期）
    cur.execute(
        """
        SELECT substr(timestamp, 1, 10) as day, COUNT(*)
        FROM events
        WHERE status = 'unsafe'
        GROUP BY day
        ORDER BY day
        """
    )
    rows_hist = cur.fetchall()
    history_labels = [r[0] for r in rows_hist]  # YYYY-MM-DD
    history_values = [r[1] for r in rows_hist]

    conn.close()

    return render_template(
        "stats.html",
        labels_today=labels_today,
        values_today=values_today,
        history_labels=history_labels,
        history_values=history_values,
    )


@app.route("/download_csv")
def download_csv():
    return send_file(CSV_PATH, as_attachment=True, download_name="violations.csv")


# ---------- API：偵測端（Pi）上報事件 ----------


@app.route("/api/events", methods=["POST"])
def api_events():
    data = request.get_json(force=True)
    required = ["timestamp", "status"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400

    # 先寫進 DB / CSV
    event_id = insert_event(data)

    # 組圖片完整網址（如果有 image_url & BASE_URL）
    base_url = os.environ.get("BASE_URL")  # 例如 https://abcd-1234-xxxx.ngrok-free.app
    full_image_url = None
    if base_url and data.get("image_url"):
        # 避免 //，簡單處理一下
        full_image_url = base_url.rstrip("/") + data["image_url"]

    # 組 LINE 訊息文字，只對 unsafe 發通知
    try:
        if data.get("status") == "unsafe":
            lines = [
                "🚨 PPE 違規事件偵測",
                f"事件 ID：{event_id}",
                f"時間：{data.get('timestamp')}",
                f"攝影機：{data.get('camera_id') or '-'}",
                f"畫面中人數：{data.get('num_people', 0)}",
                f"未戴安全帽人數：{data.get('num_no_helmet', 0)}",
            ]
            text = "\n".join(lines)
            send_line_message(text, image_url=full_image_url)
    except Exception as e:
        print(f"[WARN] Failed to send LINE notification: {e}")

    return jsonify({"status": "ok", "id": event_id})


# ---------- 會員系統：登入 / 登出 ----------


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_username(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))
        else:
            error = "帳號或密碼錯誤"

    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ---------- 管理員操作：新增 / 刪除 / 清空 ----------


def require_admin():
    if not current_user.is_authenticated or not current_user.is_admin:
        abort(403)


@app.route("/admin/events/new", methods=["GET", "POST"])
@login_required
def admin_new_event():
    require_admin()
    error = None

    # 預設值：現在時間，給 datetime-local input 用
    default_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")

    if request.method == "POST":
        try:
            # 1. 時間欄位：若有填就用使用者填的，否則用現在時間
            ts_str = request.form.get("timestamp") or ""
            if ts_str:
                # datetime-local 格式: 2025-11-30T18:30
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
                    timestamp = dt.isoformat(timespec="seconds")
                except ValueError:
                    error = "時間格式錯誤，請重新選擇。"
                    # 保留使用者剛輸入的時間
                    return render_template(
                        "event_new.html",
                        error=error,
                        default_timestamp=ts_str,
                    )
            else:
                timestamp = datetime.now().isoformat(timespec="seconds")

            camera_id = request.form.get("camera_id") or None
            num_people = int(request.form.get("num_people") or 0)
            num_no_helmet = int(request.form.get("num_no_helmet") or 0)
            image_url = request.form.get("image_url") or None

            status = "unsafe" if num_no_helmet > 0 else "safe"
            event = {
                "timestamp": timestamp,
                "camera_id": camera_id,
                "status": status,
                "num_people": num_people,
                "num_no_helmet": num_no_helmet,
                "image_url": image_url,
            }
            insert_event(event)
            return redirect(url_for("events_page"))
        except Exception as e:
            error = f"新增失敗: {e}"

    # GET 或第一次載入表單
    return render_template(
        "event_new.html",
        error=error,
        default_timestamp=default_timestamp,
    )


@app.route("/admin/events/<int:event_id>/delete", methods=["POST"])
@login_required
def admin_delete_event(event_id: int):
    require_admin()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()

    rewrite_csv_from_db()
    return redirect(url_for("events_page"))


@app.route("/admin/events/clear", methods=["POST"])
@login_required
def admin_clear_events():
    require_admin()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM events")
    conn.commit()
    conn.close()

    rewrite_csv_from_db()
    return redirect(url_for("events_page"))


if __name__ == "__main__":
    init_storage()
    app.run(host="0.0.0.0", port=5001, debug=True)
