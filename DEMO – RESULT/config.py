from __future__ import annotations

import datetime as dt
import hashlib
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtGui import QPixmap


APP_TITLE = "Система моніторингу ЗІЗ"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB_PATH = DATA / "monitoring_ziz.sqlite3"
CURRENT_MODEL = ROOT / "current_model.pt"
DB_SCHEMA_VERSION = 3
CAMERA_IDS = ("CAM_01", "CAM_02")
CAMERA_NAMES = {"CAM_01": "Камера 1", "CAM_02": "Камера 2"}

FOLDERS = {
    "uploads": DATA / "uploads",
    "fixed_new": DATA / "incidents" / "fixed_new",
    "confirmed": DATA / "incidents" / "confirmed",
    "false_positive": DATA / "incidents" / "false_positive",
    "ignored": DATA / "incidents" / "ignored",
    "deleted": DATA / "incidents" / "deleted",
    "resolved": DATA / "incidents" / "resolved",
    "raw": DATA / "raw_frames",
    "active_learning": DATA / "active_learning",
    "recovery": DATA / "recovery",
}

PERSON_CLASS = 3
HELMET_CLASS = 0
NO_HELMET_CLASS = 1
VEST_CLASS = 4
NO_VEST_CLASS = 2

CONFIDENCE = 0.25
MODEL_IMAGE_SIZE = 640
FILE_PROCESS_EVERY_N = 10
PERSON_REVIEW_COOLDOWN_SEC = 60.0
SAFE_CONFLICT_MARGIN = 0.08
TRACKER_CONFIG = "bytetrack.yaml"
VIDEO_MAX_WIDTH = 800
MODEL_SLOW_WARNING_SEC = 2.0
DISPLAY_JPEG_QUALITY = 76
LEFT_COLUMN_WIDTH = 540
VIDEO_VIEW_MIN_HEIGHT = 180
WORKER_STOP_TIMEOUT_MS = 10000
MODEL_INFERENCE_LOCK = threading.Lock()


def now_iso() -> str:
    return dt.datetime.now().isoformat(sep=" ", timespec="milliseconds")


def camera_label(camera_id: str | None) -> str:
    return CAMERA_NAMES.get(str(camera_id), str(camera_id or "Камера не визначена"))


def detection_label(label: str) -> str:
    labels = {
        "HELMET": "КАСКА",
        "VEST": "ЖИЛЕТ",
        "NO_HELMET": "БЕЗ КАСКИ",
        "NO_VEST": "БЕЗ ЖИЛЕТА",
        "PPE_OK": "ЗІЗ Є",
        "PPE_UNCLEAR": "ЗІЗ НЕ ВИЗНАЧЕНО",
        "PERSON": "ЛЮДИНА",
    }
    return " + ".join(labels.get(part, part) for part in str(label).split("+"))


def password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_storage() -> None:
    DATA.mkdir(exist_ok=True)
    for folder in FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def portable_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_absolute():
        if source.parts and source.parts[0] == ROOT.name:
            return source.as_posix()
        if source.parts and source.parts[0] == DATA.name:
            return (Path(ROOT.name) / source).as_posix()
        return source.as_posix()
    try:
        return source.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return source.as_posix()


def absolute_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    source = Path(path)
    if source.is_absolute():
        return source
    if source.parts and source.parts[0] == ROOT.name:
        return ROOT.parent / source
    return ROOT / source


def safe_move(src: str | Path | None, dst: Path) -> str | None:
    if not src:
        return None
    source = absolute_path(src)
    if source is None:
        return None
    if not source.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(f"{dst.stem}_{uuid.uuid4().hex[:6]}{dst.suffix}")
    shutil.move(str(source), str(dst))
    return portable_path(dst)


def safe_copy(src: str | Path | None, dst: Path) -> str | None:
    if not src:
        return None
    source = absolute_path(src)
    if source is None:
        return None
    if not source.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return portable_path(dst)


def write_frame_atomic(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}_{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), frame):
            raise OSError(f"Не вдалося записати кадр: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def remove_file_if_exists(path: Path | None) -> None:
    if path and path.exists():
        path.unlink()


def frame_to_jpg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def jpg_to_pixmap(data: bytes) -> QPixmap:
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap


def file_to_pixmap(path: str) -> QPixmap:
    resolved = absolute_path(path)
    pixmap = QPixmap(str(resolved) if resolved else "")
    return pixmap


def status_label(status: str) -> str:
    return {
        "pending": "Очікує рішення оператора",
        "confirmed": "Записано оператором як інцидент",
        "false_positive": "Позначено оператором як хибну тривогу",
        "ignored": "Оператор не визначив",
    }.get(status, status)


def event_type_label(event_type: str) -> str:
    return {
        "login_success": "Успішний вхід",
        "login_denied": "Невдалий вхід",
        "shift_start": "Початок зміни",
        "shift_end": "Завершення зміни",
        "decision": "Рішення щодо інциденту",
        "decision_empty": "Рішення без картки",
        "undo": "Скасування рішення",
        "shift_note": "Нотатка зміни",
        "source_start": "Початок аналізу",
        "source_start_request": "Запуск джерела",
        "source_stop": "Завершення аналізу",
        "source_stop_request": "Зупинка джерела",
        "source_stop_error": "Помилка зупинки",
        "source_error": "Помилка джерела",
        "source_blocked": "Джерело не запущено",
        "file_selected": "Обрано файл",
        "file_dialog_cancel": "Вибір файлу скасовано",
        "image_analysis_start": "Початок аналізу фото",
        "image_analysis_done": "Завершення аналізу фото",
        "model_loaded": "Модель завантажено",
        "model_error": "Помилка моделі",
        "model_slow": "Повільна обробка",
        "system_error": "Системна помилка",
        "alert": "Створено картку",
        "ppe_resolved": "ЗІЗ виправлено",
        "view_mode": "Режим рамок",
        "search_incidents": "Пошук інцидентів",
        "open_search": "Відкриття пошуку",
        "open_folder": "Відкриття папки",
        "open_frame_folder": "Розташування кадру",
        "admin_open_request": "Запит доступу адміністратора",
        "admin_open_cancel": "Доступ адміністратора скасовано",
        "admin_open_denied": "Доступ адміністратора відхилено",
        "admin_open_success": "Доступ адміністратора надано",
        "shift_finish_confirm": "Підтверджено завершення зміни",
        "shift_finish_cancel": "Завершення зміни скасовано",
        "shift_finish_start": "Почато завершення зміни",
        "shift_finish_blocked": "Завершення зміни заблоковано",
        "window_close": "Закриття вікна",
        "window_close_blocked": "Закриття вікна заблоковано",
        "app_start": "Запуск програми",
        "app_recovery": "Відновлення програми",
        "app_exit": "Вихід із програми",
        "app_exception": "Помилка програми",
        "app_thread_exception": "Помилка фонового процесу",
        "app_exec_finished": "Програму завершено",
    }.get(event_type, "Системна подія")


def display_event_message(message: str) -> str:
    replacements = (
        ("Створено картку порушення", "Створено картку перевірки"),
        ("порушення=", "висновок моделі="),
        ("оцінка ризику=", "пріоритет перевірки="),
        ("run_id=", "номер запуску="),
        ("camera_id=", "камера="),
        ("label=", "висновок моделі="),
        ("risk_score=", "пріоритет перевірки="),
        ("track_id=", "номер людини="),
        ("ppe=", "ЗІЗ="),
        ("detections=", "виявлень="),
        ("date_from=", "від="),
        ("date_to=", "до="),
        ("state=", "стан="),
        ("source=", "джерело="),
        ("results=", "знайдено="),
        ("group=", "група="),
        ("pending-картки", "картки, що очікує рішення"),
    )
    text = str(message)
    for raw, translated in replacements:
        text = text.replace(raw, translated)
    for raw in ("NO_HELMET", "NO_VEST", "HELMET", "VEST", "PPE_OK", "PPE_UNCLEAR", "PERSON"):
        text = text.replace(raw, detection_label(raw))
    return text




def person_ppe_status_label(value: Any) -> str:
    if value == "ppe_ok":
        return "ЗІЗ зафіксовано після спостереження"
    if value == "unclear":
        return "Модель не визначила стан ЗІЗ"
    return "Модель виявила ознаки порушення ЗІЗ"


def format_confidence(value: Any) -> str:
    confidence = float(value or 0.0)
    return f"{confidence:.0%} ({confidence:.2f})"


