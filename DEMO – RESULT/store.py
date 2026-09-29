from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    CAMERA_IDS,
    DB_SCHEMA_VERSION,
    FOLDERS,
    absolute_path,
    now_iso,
    password_hash,
    portable_path,
    remove_file_if_exists,
    safe_copy,
    safe_move,
    status_label,
)


@dataclass
class User:
    id: int
    login: str
    display_name: str
    is_operator: bool
    is_admin: bool


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def execute(self, sql: str, args: tuple[Any, ...] = ()) -> None:
        with self.lock, closing(self.connect()) as conn:
            conn.execute(sql, args)
            conn.commit()

    def fetchone(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self.lock, closing(self.connect()) as conn:
            return conn.execute(sql, args).fetchone()

    def fetchall(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.lock, closing(self.connect()) as conn:
            return conn.execute(sql, args).fetchall()

    def init_db(self) -> None:
        with self.lock, closing(self.connect()) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(incidents)")}
            schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if schema_version != DB_SCHEMA_VERSION or not {"original_screenshot_path", "camera_id"} <= columns:
                self.migrate_schema(conn)
            else:
                self.create_indexes(conn)
            self.seed_default_users(conn)
            conn.commit()

    @staticmethod
    def bounded_score(value: Any) -> float | None:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    def create_schema(self, conn: sqlite3.Connection) -> None:
        statements = (
            """CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_operator INTEGER NOT NULL DEFAULT 1 CHECK (is_operator IN (0, 1)),
                is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE RESTRICT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                comment TEXT
            )""",
            """CREATE TABLE login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id) ON UPDATE CASCADE ON DELETE SET NULL,
                success INTEGER NOT NULL CHECK (success IN (0, 1)),
                created_at TEXT NOT NULL,
                note TEXT
            )""",
            """CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shift_id INTEGER REFERENCES shifts(id) ON UPDATE CASCADE ON DELETE SET NULL,
                user_id INTEGER REFERENCES users(id) ON UPDATE CASCADE ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                camera_id TEXT NOT NULL CHECK (camera_id IN ('CAM_01', 'CAM_02')),
                source_file TEXT NOT NULL,
                frame_index INTEGER CHECK (frame_index IS NULL OR frame_index >= 0),
                label TEXT NOT NULL,
                person_ppe_status TEXT NOT NULL CHECK (person_ppe_status IN ('violation', 'unclear', 'ppe_ok')),
                risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
                person_confidence REAL CHECK (person_confidence IS NULL OR person_confidence BETWEEN 0 AND 1),
                helmet_confidence REAL CHECK (helmet_confidence IS NULL OR helmet_confidence BETWEEN 0 AND 1),
                vest_confidence REAL CHECK (vest_confidence IS NULL OR vest_confidence BETWEEN 0 AND 1),
                no_helmet_confidence REAL CHECK (no_helmet_confidence IS NULL OR no_helmet_confidence BETWEEN 0 AND 1),
                no_vest_confidence REAL CHECK (no_vest_confidence IS NULL OR no_vest_confidence BETWEEN 0 AND 1),
                track_id INTEGER,
                violation_key TEXT,
                ppe_summary TEXT,
                resolved_at TEXT,
                resolved_frame_index INTEGER CHECK (resolved_frame_index IS NULL OR resolved_frame_index >= 0),
                resolved_screenshot_path TEXT,
                resolved_ppe_summary TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'confirmed', 'false_positive', 'ignored', 'deleted')),
                original_screenshot_path TEXT NOT NULL,
                raw_path TEXT,
                current_file_path TEXT NOT NULL,
                active_learning_path TEXT,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by INTEGER REFERENCES users(id) ON UPDATE CASCADE ON DELETE SET NULL,
                shift_id INTEGER REFERENCES shifts(id) ON UPDATE CASCADE ON DELETE SET NULL
            )""",
            """CREATE TABLE decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL REFERENCES incidents(id) ON UPDATE CASCADE ON DELETE CASCADE,
                previous_status TEXT NOT NULL
                    CHECK (previous_status IN ('pending', 'confirmed', 'false_positive', 'ignored', 'deleted')),
                new_status TEXT NOT NULL
                    CHECK (new_status IN ('pending', 'confirmed', 'false_positive', 'ignored', 'deleted')),
                previous_file_path TEXT,
                new_file_path TEXT,
                active_learning_path TEXT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE RESTRICT,
                shift_id INTEGER REFERENCES shifts(id) ON UPDATE CASCADE ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0 CHECK (undone IN (0, 1))
            )""",
        )
        for statement in statements:
            conn.execute(statement)
        self.create_indexes(conn)

    @staticmethod
    def create_indexes(conn: sqlite3.Connection) -> None:
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_status_created ON incidents(status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_ppe_created ON incidents(person_ppe_status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_source_created ON incidents(source_file, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_camera_created ON incidents(camera_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_login_attempts_created_at ON login_attempts(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_incident_created ON decisions(incident_id, created_at DESC)",
        ):
            conn.execute(statement)

    def migrate_schema(self, conn: sqlite3.Connection) -> None:
        tables = ("users", "shifts", "login_attempts", "events", "incidents", "decisions")
        previous: dict[str, list[dict[str, Any]]] = {}
        for table in tables:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            previous[table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")] if exists else []

        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            for table in reversed(tables):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            self.create_schema(conn)

            for row in previous["users"]:
                conn.execute(
                    """
                    INSERT INTO users (id, login, display_name, password_hash, is_operator, is_admin, active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["login"],
                        row["display_name"],
                        row["password_hash"],
                        int(bool(row.get("is_operator", 1))),
                        int(bool(row.get("is_admin", 0))),
                        int(bool(row.get("active", 1))),
                        row["created_at"],
                    ),
                )
            valid_users = {row["id"] for row in previous["users"]}
            for row in previous["shifts"]:
                if row.get("user_id") not in valid_users:
                    continue
                conn.execute(
                    "INSERT INTO shifts (id, user_id, started_at, ended_at, comment) VALUES (?, ?, ?, ?, ?)",
                    (row["id"], row["user_id"], row["started_at"], row.get("ended_at"), row.get("comment")),
                )
            valid_shifts = {row["id"] for row in previous["shifts"] if row.get("user_id") in valid_users}
            user_by_login = {row["login"]: row["id"] for row in previous["users"]}
            for row in previous["login_attempts"]:
                conn.execute(
                    """
                    INSERT INTO login_attempts (id, login, user_id, success, created_at, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["login"],
                        row.get("user_id") if row.get("user_id") in valid_users else user_by_login.get(row["login"]),
                        int(bool(row["success"])),
                        row["created_at"],
                        row.get("note"),
                    ),
                )
            for row in previous["events"]:
                conn.execute(
                    """
                    INSERT INTO events (id, shift_id, user_id, event_type, message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row.get("shift_id") if row.get("shift_id") in valid_shifts else None,
                        row.get("user_id") if row.get("user_id") in valid_users else None,
                        row["event_type"],
                        row["message"],
                        row["created_at"],
                    ),
                )
            valid_incidents: set[int] = set()
            for row in previous["incidents"]:
                legacy_score = self.bounded_score(row.get("risk_score", row.get("confidence", 0.0))) or 0.0
                label = str(row.get("label") or "PPE_UNCLEAR")
                ppe_status = row.get("person_ppe_status")
                if ppe_status not in {"violation", "unclear", "ppe_ok"}:
                    if row.get("ppe_resolution") == "equipped":
                        ppe_status = "ppe_ok"
                    elif label == "PPE_UNCLEAR":
                        ppe_status = "unclear"
                    else:
                        ppe_status = "violation"
                original_path = row.get("original_screenshot_path") or row.get("screenshot_path") or row.get("current_file_path") or ""
                no_helmet = row.get("no_helmet_confidence")
                no_vest = row.get("no_vest_confidence")
                if "NO_HELMET" in label and no_helmet is None:
                    no_helmet = legacy_score
                if "NO_VEST" in label and no_vest is None:
                    no_vest = legacy_score
                conn.execute(
                    """
                    INSERT INTO incidents
                    (id, uid, camera_id, source_file, frame_index, label, person_ppe_status, risk_score,
                     person_confidence, helmet_confidence, vest_confidence, no_helmet_confidence, no_vest_confidence,
                     track_id, violation_key, ppe_summary, resolved_at, resolved_frame_index,
                     resolved_screenshot_path, resolved_ppe_summary, status, original_screenshot_path,
                     raw_path, current_file_path, active_learning_path, created_at, decided_at, decided_by, shift_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["uid"],
                        row.get("camera_id") if row.get("camera_id") in CAMERA_IDS else "CAM_01",
                        row.get("source_file") or "невідоме джерело",
                        row.get("frame_index"),
                        label,
                        ppe_status,
                        legacy_score,
                        self.bounded_score(row.get("person_confidence")),
                        self.bounded_score(row.get("helmet_confidence")),
                        self.bounded_score(row.get("vest_confidence")),
                        self.bounded_score(no_helmet),
                        self.bounded_score(no_vest),
                        row.get("track_id"),
                        row.get("violation_key"),
                        row.get("ppe_summary"),
                        row.get("resolved_at"),
                        row.get("resolved_frame_index"),
                        portable_path(row.get("resolved_screenshot_path")),
                        row.get("resolved_ppe_summary"),
                        row.get("status") if row.get("status") in {"pending", "confirmed", "false_positive", "ignored", "deleted"} else "pending",
                        portable_path(original_path) or "",
                        portable_path(row.get("raw_path")),
                        portable_path(row.get("current_file_path") or original_path) or "",
                        portable_path(row.get("active_learning_path")),
                        row["created_at"],
                        row.get("decided_at"),
                        row.get("decided_by") if row.get("decided_by") in valid_users else None,
                        row.get("shift_id") if row.get("shift_id") in valid_shifts else None,
                    ),
                )
                valid_incidents.add(row["id"])
            for row in previous["decisions"]:
                if row.get("incident_id") not in valid_incidents or row.get("user_id") not in valid_users:
                    continue
                conn.execute(
                    """
                    INSERT INTO decisions
                    (id, incident_id, previous_status, new_status, previous_file_path, new_file_path,
                     active_learning_path, user_id, shift_id, created_at, undone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["incident_id"],
                        row["previous_status"],
                        row["new_status"],
                        portable_path(row.get("previous_file_path")),
                        portable_path(row.get("new_file_path")),
                        portable_path(row.get("active_learning_path")),
                        row["user_id"],
                        row.get("shift_id") if row.get("shift_id") in valid_shifts else None,
                        row["created_at"],
                        int(bool(row.get("undone", 0))),
                    ),
                )
            conn.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError("Помилка міграції: порушено зв'язки SQLite.")

    @staticmethod
    def seed_default_users(conn: sqlite3.Connection) -> None:
        defaults = [
            ("operator_a", "Оператор А", "0000000", 1, 1),
            ("operator_b", "Оператор Б", "0000000", 1, 0),
            ("operator_v", "Оператор В", "0000000", 1, 0),
            ("operator_g", "Оператор Г", "0000000", 1, 0),
        ]
        for login, name, password, is_operator, is_admin in defaults:
            conn.execute(
                """
                INSERT OR IGNORE INTO users
                (login, display_name, password_hash, is_operator, is_admin, active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (login, name, password_hash(password), is_operator, is_admin, now_iso()),
            )

    def log_event(self, shift_id: int | None, user_id: int | None, event_type: str, message: str) -> None:
        self.execute(
            "INSERT INTO events (shift_id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (shift_id, user_id, event_type, message, now_iso()),
        )

    def login(self, login: str, password: str) -> User | None:
        login = login.strip()
        row = self.fetchone("SELECT * FROM users WHERE login = ? AND active = 1", (login,))
        success = bool(row and row["password_hash"] == password_hash(password))
        self.execute(
            "INSERT INTO login_attempts (login, user_id, success, created_at, note) VALUES (?, ?, ?, ?, ?)",
            (login, row["id"] if row else None, int(success), now_iso(), "Успішний вхід" if success else "Невдала спроба входу"),
        )
        if success and row:
            self.log_event(None, row["id"], "login_success", f"Успішний вхід: {login}.")
        else:
            self.log_event(None, row["id"] if row else None, "login_denied", f"Невдала спроба входу: {login or '(порожній логін)'}.")
        if not success or row is None:
            return None
        return User(row["id"], row["login"], row["display_name"], bool(row["is_operator"]), bool(row["is_admin"]))

    def verify_admin_password(self, password: str) -> bool:
        rows = self.fetchall("SELECT password_hash FROM users WHERE is_admin = 1 AND active = 1")
        return any(row["password_hash"] == password_hash(password) for row in rows)

    def start_shift(self, user: User) -> int:
        with self.lock, closing(self.connect()) as conn:
            cur = conn.execute("INSERT INTO shifts (user_id, started_at) VALUES (?, ?)", (user.id, now_iso()))
            last_id = cur.lastrowid
            if last_id is None:
                raise RuntimeError("Не вдалося отримати id нової зміни.")
            shift_id = last_id
            conn.commit()
        self.log_event(shift_id, user.id, "shift_start", f"Зміну прийняв: {user.display_name}")
        return shift_id

    def end_shift(self, shift_id: int, user_id: int, comment: str | None = None) -> bool:
        ended_at = now_iso()
        with self.lock, closing(self.connect()) as conn:
            row = conn.execute("SELECT ended_at FROM shifts WHERE id = ?", (shift_id,)).fetchone()
            if row is None or row["ended_at"]:
                return False
            if comment is None:
                conn.execute("UPDATE shifts SET ended_at = ? WHERE id = ?", (ended_at, shift_id))
            else:
                conn.execute("UPDATE shifts SET ended_at = ?, comment = ? WHERE id = ?", (ended_at, comment, shift_id))
            conn.execute(
                "INSERT INTO events (shift_id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (shift_id, user_id, "shift_end", "Зміну завершено", ended_at),
            )
            conn.commit()
        return True

    def add_shift_note(self, shift_id: int, user: User, note: str) -> bool:
        note = note.strip()
        if not note:
            return False
        timestamp = now_iso()
        entry = f"[{timestamp}] {user.display_name}: {note}"
        with self.lock, closing(self.connect()) as conn:
            row = conn.execute("SELECT comment FROM shifts WHERE id = ? AND ended_at IS NULL", (shift_id,)).fetchone()
            if row is None:
                return False
            previous = (row["comment"] or "").strip()
            updated = f"{previous}\n{entry}" if previous else entry
            conn.execute("UPDATE shifts SET comment = ? WHERE id = ?", (updated, shift_id))
            conn.execute(
                "INSERT INTO events (shift_id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (shift_id, user.id, "shift_note", "Додано нотатку до активної зміни.", timestamp),
            )
            conn.commit()
        return True

    def recent_shift_notes(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT s.id, s.started_at, s.ended_at, s.comment, u.display_name
            FROM shifts s
            JOIN users u ON u.id = s.user_id
            WHERE trim(COALESCE(s.comment, '')) != ''
            ORDER BY s.started_at DESC, s.id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def recover_open_shifts(self) -> int:
        recovered_at = now_iso()
        note = "Системне відновлення: попередній запуск завершився без коректного закриття зміни."
        with self.lock, closing(self.connect()) as conn:
            rows = conn.execute("SELECT id, user_id, comment FROM shifts WHERE ended_at IS NULL ORDER BY id").fetchall()
            for row in rows:
                comment = (row["comment"] or "").strip()
                recovered_comment = f"{comment}\n{note}" if comment else note
                conn.execute(
                    "UPDATE shifts SET ended_at = ?, comment = ? WHERE id = ? AND ended_at IS NULL",
                    (recovered_at, recovered_comment, row["id"]),
                )
                conn.execute(
                    "INSERT INTO events (shift_id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (row["id"], row["user_id"], "app_recovery", note, recovered_at),
                )
                conn.execute(
                    "INSERT INTO events (shift_id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (row["id"], row["user_id"], "shift_end", "Зміну завершено автоматично після некоректного закриття програми.", recovered_at),
                )
            conn.commit()
        return len(rows)

    def recover_file_consistency(self) -> tuple[int, int]:
        status_folder = {
            "pending": "fixed_new",
            "confirmed": "confirmed",
            "false_positive": "false_positive",
            "ignored": "ignored",
            "deleted": "deleted",
        }
        restored = 0
        isolated = 0
        with self.lock, closing(self.connect()) as conn:
            rows = conn.execute("SELECT id, status, current_file_path FROM incidents").fetchall()
            for row in rows:
                current_path = absolute_path(row["current_file_path"])
                if current_path is None or current_path.exists():
                    continue
                expected = FOLDERS[status_folder.get(row["status"], "fixed_new")] / current_path.name
                candidates = [
                    FOLDERS[name] / current_path.name
                    for name in ("fixed_new", "confirmed", "false_positive", "ignored", "deleted")
                ]
                found = next((candidate for candidate in candidates if candidate.exists()), None)
                if found is None:
                    continue
                recovered_path = safe_move(found, expected) if found != expected else portable_path(found)
                if recovered_path:
                    conn.execute(
                        "UPDATE incidents SET current_file_path = ? WHERE id = ?",
                        (recovered_path, row["id"]),
                    )
                    restored += 1

            referenced: set[Path] = set()
            for row in conn.execute(
                "SELECT current_file_path, raw_path, resolved_screenshot_path, active_learning_path FROM incidents"
            ):
                for column in ("current_file_path", "raw_path", "resolved_screenshot_path", "active_learning_path"):
                    path = absolute_path(row[column])
                    if path is not None:
                        referenced.add(path.resolve())
            for folder_name in ("fixed_new", "raw", "resolved", "active_learning"):
                for file_path in FOLDERS[folder_name].glob("*"):
                    if not file_path.is_file() or file_path.resolve() in referenced:
                        continue
                    safe_move(file_path, FOLDERS["recovery"] / file_path.name)
                    isolated += 1
            conn.commit()
        return restored, isolated

    def add_incident(
        self,
        camera_id: str,
        source_file: str,
        frame_index: int,
        label: str,
        person_ppe_status: str,
        risk_score: float,
        original_screenshot_path: str,
        raw_path: str | None,
        shift_id: int | None,
        track_id: int | None = None,
        violation_key: str | None = None,
        ppe_summary: str | None = None,
        person_confidence: float | None = None,
        helmet_confidence: float | None = None,
        vest_confidence: float | None = None,
        no_helmet_confidence: float | None = None,
        no_vest_confidence: float | None = None,
    ) -> int:
        if camera_id not in CAMERA_IDS:
            raise ValueError(f"Невідома камера: {camera_id}.")
        if person_ppe_status == "ppe_ok" or label == "PPE_OK":
            raise ValueError("Картка для PPE_OK не створюється.")
        uid = uuid.uuid4().hex
        original_screenshot_path = portable_path(original_screenshot_path) or ""
        raw_path = portable_path(raw_path)
        with self.lock, closing(self.connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO incidents
                (uid, camera_id, source_file, frame_index, label, person_ppe_status, risk_score,
                 person_confidence, helmet_confidence, vest_confidence, no_helmet_confidence, no_vest_confidence,
                 track_id, violation_key, ppe_summary, status, original_screenshot_path,
                 raw_path, current_file_path, created_at, shift_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    camera_id,
                    source_file,
                    frame_index,
                    label,
                    person_ppe_status,
                    self.bounded_score(risk_score) or 0.0,
                    self.bounded_score(person_confidence),
                    self.bounded_score(helmet_confidence),
                    self.bounded_score(vest_confidence),
                    self.bounded_score(no_helmet_confidence),
                    self.bounded_score(no_vest_confidence),
                    track_id,
                    violation_key,
                    ppe_summary,
                    original_screenshot_path,
                    raw_path,
                    original_screenshot_path,
                    now_iso(),
                    shift_id,
                ),
            )
            last_id = cur.lastrowid
            if last_id is None:
                raise RuntimeError("Не вдалося отримати id нового інциденту.")
            incident_id = last_id
            conn.commit()
        return incident_id

    def mark_incident_equipped(
        self,
        incident_id: int,
        frame_index: int,
        screenshot_path: str,
        ppe_summary: str | None,
    ) -> bool:
        row = self.fetchone("SELECT person_ppe_status FROM incidents WHERE id = ?", (incident_id,))
        if row is None or row["person_ppe_status"] == "ppe_ok":
            return False
        with self.lock, closing(self.connect()) as conn:
            cur = conn.execute(
                """
                UPDATE incidents
                SET person_ppe_status = 'ppe_ok',
                    resolved_at = ?,
                    resolved_frame_index = ?,
                    resolved_screenshot_path = ?,
                    resolved_ppe_summary = ?
                WHERE id = ?
                  AND person_ppe_status != 'ppe_ok'
                """,
                (now_iso(), frame_index, portable_path(screenshot_path), ppe_summary, incident_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def pending_incident(self, order: str) -> sqlite3.Row | None:
        direction = "DESC" if order == "latest" else "ASC"
        return self.fetchone(
            f"SELECT * FROM incidents WHERE status = 'pending' ORDER BY created_at {direction}, id {direction} LIMIT 1"
        )

    def decide(self, incident_id: int, status: str, user_id: int, shift_id: int | None) -> str:
        with self.lock, closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if row is None:
                return "Картку не знайдено."
            if row["status"] != "pending":
                return "Картка вже має рішення."

            target_file = FOLDERS[status] / Path(row["current_file_path"]).name
            new_file = safe_move(row["current_file_path"], target_file) or row["current_file_path"]
            active_learning_path = None
            try:
                if status == "false_positive":
                    source = row["raw_path"] or new_file
                    suffix = Path(source).suffix if source else ".jpg"
                    active_learning_path = safe_copy(source, FOLDERS["active_learning"] / f"{Path(new_file).stem}_raw{suffix}")
                changed = conn.execute(
                    """
                    UPDATE incidents
                    SET status = ?, current_file_path = ?, active_learning_path = ?,
                        decided_at = ?, decided_by = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (status, new_file, active_learning_path, now_iso(), user_id, incident_id),
                )
                if changed.rowcount != 1:
                    raise RuntimeError("Стан картки змінився під час запису рішення.")
                conn.execute(
                    """
                    INSERT INTO decisions
                    (incident_id, previous_status, new_status, previous_file_path, new_file_path,
                     active_learning_path, user_id, shift_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (incident_id, row["status"], status, row["current_file_path"], new_file, active_learning_path, user_id, shift_id, now_iso()),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                remove_file_if_exists(absolute_path(active_learning_path))
                if new_file != row["current_file_path"]:
                    safe_move(new_file, absolute_path(row["current_file_path"]) or FOLDERS["fixed_new"] / Path(new_file).name)
                raise

        self.log_event(shift_id, user_id, "decision", f"Інцидент #{incident_id}: {status_label(status)}")
        return "Рішення записано."

    def undo_last_decision(self, user_id: int, shift_id: int | None) -> str:
        shift_filter = "AND d.shift_id = ?" if shift_id is not None else "AND d.shift_id IS NULL"
        args: tuple[Any, ...] = (user_id, shift_id) if shift_id is not None else (user_id,)
        decision = self.fetchone(
            f"""
            SELECT d.*
            FROM decisions d
            JOIN incidents i ON i.id = d.incident_id
            WHERE d.undone = 0
              AND d.user_id = ?
              {shift_filter}
              AND i.status = d.new_status
            ORDER BY d.created_at DESC, d.id DESC
            LIMIT 1
            """,
            args,
        )
        if decision is None:
            return "Немає вашого рішення в поточній зміні для відміни."

        with self.lock, closing(self.connect()) as conn:
            current_path = decision["new_file_path"]
            restore_name = Path(decision["previous_file_path"]).name if decision["previous_file_path"] else Path(current_path).name
            restored_path = safe_move(current_path, FOLDERS["fixed_new"] / restore_name) or portable_path(FOLDERS["fixed_new"] / restore_name)
            learning_path = absolute_path(decision["active_learning_path"])
            try:
                conn.execute(
                    """
                    UPDATE incidents
                    SET status = 'pending', current_file_path = ?, active_learning_path = NULL,
                        decided_at = NULL, decided_by = NULL
                    WHERE id = ?
                    """,
                    (restored_path, decision["incident_id"]),
                )
                conn.execute("UPDATE decisions SET undone = 1 WHERE id = ?", (decision["id"],))
                conn.commit()
            except Exception:
                conn.rollback()
                if restored_path and restored_path != current_path:
                    safe_move(restored_path, absolute_path(current_path) or FOLDERS["fixed_new"] / Path(restored_path).name)
                raise
            remove_file_if_exists(learning_path)
        self.log_event(shift_id, user_id, "undo", f"Відмінено рішення для інциденту #{decision['incident_id']}")
        return "Останнє рішення скасовано: картку повернено до черги очікування."

    def stats(self) -> dict[str, Any]:
        rows = self.fetchall("SELECT status, COUNT(*) AS n, AVG(risk_score) AS avg_risk FROM incidents GROUP BY status")
        data = {row["status"]: {"n": row["n"], "avg_risk": row["avg_risk"] or 0.0} for row in rows}
        total = sum(item["n"] for item in data.values())
        reviewed = data.get("confirmed", {"n": 0})["n"] + data.get("false_positive", {"n": 0})["n"] + data.get("ignored", {"n": 0})["n"]
        avg_risk = self.fetchone("SELECT AVG(risk_score) AS value FROM incidents")
        confirmed_risk = self.fetchone("SELECT AVG(risk_score) AS value FROM incidents WHERE status = 'confirmed'")
        false_positive = data.get("false_positive", {"n": 0})["n"]
        return {
            "total": total,
            "pending": data.get("pending", {"n": 0})["n"],
            "confirmed": data.get("confirmed", {"n": 0})["n"],
            "false_positive": false_positive,
            "ignored": data.get("ignored", {"n": 0})["n"],
            "reviewed": reviewed,
            "false_positive_rate": false_positive / reviewed if reviewed else 0.0,
            "avg_risk": (avg_risk["value"] or 0.0) if avg_risk else 0.0,
            "confirmed_avg_risk": (confirmed_risk["value"] or 0.0) if confirmed_risk else 0.0,
        }

    def search_incidents(
        self,
        date_from: str = "",
        date_to: str = "",
        state_filter: str = "all",
        camera_id: str = "",
        source: str = "",
    ) -> list[sqlite3.Row]:
        filters = []
        args: list[Any] = []
        if date_from:
            filters.append("date(i.created_at) >= date(?)")
            args.append(date_from)
        if date_to:
            filters.append("date(i.created_at) <= date(?)")
            args.append(date_to)
        if state_filter.startswith("status:"):
            filters.append("i.status = ?")
            args.append(state_filter.split(":", 1)[1])
        elif state_filter.startswith("ppe:"):
            filters.append("i.person_ppe_status = ?")
            args.append(state_filter.split(":", 1)[1])
        if camera_id in CAMERA_IDS:
            filters.append("i.camera_id = ?")
            args.append(camera_id)
        if source:
            filters.append("i.source_file = ?")
            args.append(source)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        return self.fetchall(
            f"""
            SELECT
                i.*,
                s.started_at AS shift_started_at,
                s.ended_at AS shift_ended_at,
                s.comment AS shift_comment,
                shift_user.display_name AS shift_operator_name,
                decision_user.display_name AS decision_user_name,
                d.created_at AS last_decision_at,
                d.previous_status AS last_previous_status,
                d.new_status AS last_new_status
            FROM incidents i
            LEFT JOIN shifts s ON s.id = i.shift_id
            LEFT JOIN users shift_user ON shift_user.id = s.user_id
            LEFT JOIN users decision_user ON decision_user.id = i.decided_by
            LEFT JOIN decisions d ON d.id = (
                SELECT d2.id
                FROM decisions d2
                WHERE d2.incident_id = i.id AND d2.undone = 0
                ORDER BY d2.created_at DESC, d2.id DESC
                LIMIT 1
            )
            {where}
            ORDER BY i.created_at DESC, i.id DESC
            """,
            tuple(args),
        )

    def incident_sources(self, camera_id: str = "") -> list[str]:
        if camera_id in CAMERA_IDS:
            rows = self.fetchall(
                "SELECT DISTINCT source_file FROM incidents WHERE camera_id = ? AND source_file != '' ORDER BY source_file COLLATE NOCASE",
                (camera_id,),
            )
        else:
            rows = self.fetchall(
                "SELECT DISTINCT source_file FROM incidents WHERE source_file != '' ORDER BY source_file COLLATE NOCASE"
            )
        return [str(row["source_file"]) for row in rows]

    def recent_events(self, limit: int = 40) -> list[sqlite3.Row]:
        return self.fetchall("SELECT * FROM events ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))

    def list_users(self) -> list[sqlite3.Row]:
        return self.fetchall("SELECT * FROM users ORDER BY active DESC, id ASC")

    def save_user(self, login: str, name: str, password: str, is_operator: bool, is_admin: bool) -> str:
        login = login.strip()
        name = name.strip()
        if not login or not name:
            return "Не зареєстровано: заповніть логін та ім'я."
        if not is_operator and not is_admin:
            return "Не зареєстровано: оберіть роль користувача."
        with self.lock, closing(self.connect()) as conn:
            existing = conn.execute("SELECT * FROM users WHERE login = ?", (login,)).fetchone()
            if existing is None and not password:
                return "Не зареєстровано: введіть пароль для нового користувача."
            if existing and existing["is_admin"] and not is_admin:
                active_admins = conn.execute(
                    "SELECT COUNT(*) AS n FROM users WHERE is_admin = 1 AND active = 1 AND id != ?",
                    (existing["id"],),
                ).fetchone()
                if active_admins and int(active_admins["n"]) == 0:
                    return "Не збережено: це останній активний адміністратор."
            if existing:
                if password:
                    conn.execute(
                        """
                        UPDATE users
                        SET display_name = ?, is_operator = ?, is_admin = ?, password_hash = ?, active = 1
                        WHERE login = ?
                        """,
                        (name, int(is_operator), int(is_admin), password_hash(password), login),
                    )
                else:
                    conn.execute(
                        "UPDATE users SET display_name = ?, is_operator = ?, is_admin = ?, active = 1 WHERE login = ?",
                        (name, int(is_operator), int(is_admin), login),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO users
                    (login, display_name, password_hash, is_operator, is_admin, active, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (login, name, password_hash(password), int(is_operator), int(is_admin), now_iso()),
                )
            conn.commit()
        return "Користувача збережено."

    def deactivate_user(self, user_id: int) -> str:
        row = self.fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
        if row is None:
            return "Не деактивовано: користувача не знайдено."
        if not row["active"]:
            return "Користувач уже деактивований."
        if row["is_admin"]:
            active_admins = self.fetchone(
                "SELECT COUNT(*) AS n FROM users WHERE is_admin = 1 AND active = 1 AND id != ?",
                (user_id,),
            )
            if active_admins and int(active_admins["n"]) == 0:
                return "Не деактивовано: це останній активний адміністратор."
        self.execute("UPDATE users SET active = 0 WHERE id = ?", (user_id,))
        return "Користувача деактивовано: запис лишився у БД, але вхід для нього вимкнено."


