from __future__ import annotations

import datetime as dt
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QDate, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_TITLE,
    CAMERA_IDS,
    DATA,
    FOLDERS,
    LEFT_COLUMN_WIDTH,
    ROOT,
    VIDEO_VIEW_MIN_HEIGHT,
    WORKER_STOP_TIMEOUT_MS,
    absolute_path,
    camera_label,
    detection_label,
    display_event_message,
    event_type_label,
    file_to_pixmap,
    format_confidence,
    frame_to_jpg,
    jpg_to_pixmap,
    person_ppe_status_label,
    remove_file_if_exists,
    status_label,
    write_frame_atomic,
)
from detector import Detector, VideoWorker
from store import Store, User


def show_message(parent: QWidget, title: str, message: str, icon: QMessageBox.Icon) -> None:
    dialog = QMessageBox(parent)
    dialog.setIcon(icon)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    close_btn = dialog.addButton("Гаразд", QMessageBox.ButtonRole.AcceptRole)
    dialog.setDefaultButton(close_btn)
    dialog.exec()


def show_warning(parent: QWidget, title: str, message: str) -> None:
    show_message(parent, title, message, QMessageBox.Icon.Warning)


def show_information(parent: QWidget, title: str, message: str) -> None:
    show_message(parent, title, message, QMessageBox.Icon.Information)


def ask_yes_no(parent: QWidget, title: str, message: str) -> bool:
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Question)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    tak_btn = dialog.addButton("Так", QMessageBox.ButtonRole.YesRole)
    ni_btn = dialog.addButton("Ні", QMessageBox.ButtonRole.NoRole)
    dialog.setDefaultButton(ni_btn)
    dialog.setEscapeButton(ni_btn)
    dialog.exec()
    return dialog.clickedButton() is tak_btn




class LoginDialog(QDialog):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.user: User | None = None
        self.setWindowTitle(APP_TITLE)
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        subtitle = QLabel("Вхід оператора")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        self.login_input = QLineEdit("operator_a")
        self.login_input.setPlaceholderText("Логін")
        self.password_input = QLineEdit("0000000")
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.error_label = QLabel("")
        self.error_label.setObjectName("error")

        login_button = QPushButton("УВІЙТИ")
        login_button.clicked.connect(self.try_login)
        login_button.setDefault(True)
        self.password_input.returnPressed.connect(self.try_login)

        layout.addWidget(self.login_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.error_label)
        layout.addWidget(login_button)
        layout.addWidget(QLabel("Демо: operator_a / 0000000"))

    def try_login(self) -> None:
        user = self.store.login(self.login_input.text(), self.password_input.text())
        if user is None:
            self.error_label.setText("Неправильний логін або пароль. Спробу записано.")
            return
        self.user = user
        self.accept()


class IncidentPanel(QFrame):
    def __init__(self, title: str, on_decide, on_undo, on_refresh, order: str):
        super().__init__()
        self.order = order
        self.on_decide = on_decide
        self.on_undo = on_undo
        self.on_refresh = on_refresh
        self.current_id: int | None = None
        self.setObjectName("panel")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title))
        self.title_label = QLabel("Черга порожня")
        self.image_label = QLabel("Немає кадру")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(260)
        self.image_label.setObjectName("incidentImage")

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

        row1 = QHBoxLayout()
        hybna_tryvoha_btn = QPushButton("Ні, хибна тривога")
        zapysaty_incydent_btn = QPushButton("Так, записати інцидент")
        hybna_tryvoha_btn.clicked.connect(lambda: self.on_decide(self.order, "false_positive"))
        zapysaty_incydent_btn.clicked.connect(lambda: self.on_decide(self.order, "confirmed"))
        row1.addWidget(hybna_tryvoha_btn)
        row1.addWidget(zapysaty_incydent_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        ne_vyznachaty_btn = QPushButton("Не визначати")
        skasuvaty_rishennia_btn = QPushButton("Скасувати останнє рішення")
        ne_vyznachaty_btn.clicked.connect(lambda: self.on_decide(self.order, "ignored"))
        skasuvaty_rishennia_btn.clicked.connect(self.on_undo)
        row2.addWidget(ne_vyznachaty_btn)
        row2.addWidget(skasuvaty_rishennia_btn)
        layout.addLayout(row2)

    def set_incident(self, row: sqlite3.Row | None) -> None:
        if row is None:
            self.current_id = None
            self.title_label.setText("Черга порожня")
            self.image_label.setText("Немає кадру")
            self.image_label.setPixmap(QPixmap())
            return

        self.current_id = row["id"]
        track_text = str(row["track_id"]) if row["track_id"] is not None else "не визначено"
        self.title_label.setText(f"Інцидент #{row['id']} | Номер людини: {track_text}")
        path = row["current_file_path"]
        resolved = absolute_path(path)
        if resolved and resolved.exists():
            pixmap = file_to_pixmap(path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.image_label.setText("Файл кадру не знайдено")


class IncidentSearchPanel(QWidget):
    FILTER_OPTIONS = [
        ("Усі рішення / висновки моделі", "all"),
        ("Рішення оператора: очікує", "status:pending"),
        ("Рішення оператора: записати інцидент", "status:confirmed"),
        ("Рішення оператора: хибна тривога", "status:false_positive"),
        ("Рішення оператора: не визначено", "status:ignored"),
        ("Висновок моделі: ознаки порушення ЗІЗ", "ppe:violation"),
        ("Висновок моделі: стан ЗІЗ не визначено", "ppe:unclear"),
        ("Висновок моделі: ЗІЗ зафіксовано пізніше", "ppe:ppe_ok"),
    ]

    def __init__(self, store: Store, parent=None):
        super().__init__(parent)
        self.store = store
        self.logger = getattr(parent, "log_action", None)
        self.rows: list[sqlite3.Row] = []

        layout = QVBoxLayout(self)
        filters = QGridLayout()

        self.date_from = QDateEdit(QDate.currentDate())
        self.date_to = QDateEdit(QDate.currentDate())
        for date_picker in (self.date_from, self.date_to):
            date_picker.setCalendarPopup(True)
            date_picker.setDisplayFormat("yyyy-MM-dd")
        self.state_filter = QComboBox()
        for label, value in self.FILTER_OPTIONS:
            self.state_filter.addItem(label, value)
        self.camera_filter = QComboBox()
        self.camera_filter.addItem("Усі камери", "")
        for camera_id in CAMERA_IDS:
            self.camera_filter.addItem(camera_label(camera_id), camera_id)
        self.camera_filter.currentIndexChanged.connect(self.refresh_sources)
        self.source_filter = QComboBox()
        self.refresh_sources()

        filters.addWidget(QLabel("Дата від"), 0, 0)
        filters.addWidget(self.date_from, 0, 1)
        filters.addWidget(QLabel("Дата до"), 0, 2)
        filters.addWidget(self.date_to, 0, 3)
        filters.addWidget(QLabel("Рішення / висновок"), 1, 0)
        filters.addWidget(self.state_filter, 1, 1)
        filters.addWidget(QLabel("Джерело"), 1, 2)
        filters.addWidget(self.source_filter, 1, 3)
        filters.addWidget(QLabel("Камера"), 2, 0)
        filters.addWidget(self.camera_filter, 2, 1)

        search_btn = QPushButton("Пошук")
        search_btn.clicked.connect(self.run_search)
        layout.addLayout(filters)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        open_folder_btn = QPushButton("Відкрити розташування кадру")
        open_folder_btn.clicked.connect(self.open_selected_folder)
        action_row.addWidget(search_btn)
        action_row.addWidget(open_folder_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        content = QHBoxLayout()
        self.results = QListWidget()
        self.results.currentRowChanged.connect(self.show_selected)
        content.addWidget(self.results, 2)

        details_layout = QVBoxLayout()
        self.image_label = QLabel("Оберіть інцидент")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(340, 200)
        self.image_label.setObjectName("incidentImage")
        details_layout.addWidget(self.image_label)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        details_layout.addWidget(self.details, 1)

        content.addLayout(details_layout, 3)
        layout.addLayout(content, 1)

        self.run_search()

    def log_action(self, event_type: str, message: str) -> None:
        if callable(self.logger):
            self.logger(event_type, message)

    def refresh_sources(self, *_args) -> None:
        selected = self.source_filter.currentData() if self.source_filter.count() else ""
        self.source_filter.clear()
        self.source_filter.addItem("Усі джерела", "")
        camera_id = str(self.camera_filter.currentData() or "")
        for source in self.store.incident_sources(camera_id):
            self.source_filter.addItem(source, source)
        index = self.source_filter.findData(selected)
        self.source_filter.setCurrentIndex(index if index >= 0 else 0)

    def run_search(self) -> None:
        self.refresh_sources()
        date_from = self.date_from.date().toString("yyyy-MM-dd")
        date_to = self.date_to.date().toString("yyyy-MM-dd")
        if self.date_from.date() > self.date_to.date():
            show_warning(self, "Пошук інцидентів", "Дата від не може бути пізнішою за дату до.")
            return

        self.rows = self.store.search_incidents(
            date_from=date_from,
            date_to=date_to,
            state_filter=str(self.state_filter.currentData()),
            camera_id=str(self.camera_filter.currentData() or ""),
            source=str(self.source_filter.currentData() or ""),
        )
        self.log_action(
            "search_incidents",
            f"Пошук інцидентів: від={date_from or '-'}, до={date_to or '-'}, "
            f"стан={self.state_filter.currentText()}, "
            f"камера={self.camera_filter.currentText()}, джерело={self.source_filter.currentText()}, знайдено={len(self.rows)}",
        )
        self.results.clear()
        for row in self.rows:
            self.results.addItem(self.summary_text(row))
        if self.rows:
            self.results.setCurrentRow(0)
        else:
            self.image_label.setText("Інциденти не знайдені")
            self.image_label.setPixmap(QPixmap())
            self.details.setPlainText("За заданими фільтрами немає записів.")

    def current_row(self) -> sqlite3.Row | None:
        index = self.results.currentRow()
        if index < 0 or index >= len(self.rows):
            return None
        return self.rows[index]

    def show_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.rows):
            return
        row = self.rows[index]
        self.details.setPlainText(self.details_text(row))
        path = self.display_frame_path(row)
        resolved = absolute_path(path)
        if resolved and resolved.exists():
            pixmap = file_to_pixmap(path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Файл кадру не знайдено")

    def display_frame_path(self, row: sqlite3.Row) -> str:
        if row["person_ppe_status"] == "ppe_ok" and row["resolved_screenshot_path"]:
            return str(row["resolved_screenshot_path"])
        return str(row["current_file_path"])

    def summary_text(self, row: sqlite3.Row) -> str:
        operator = row["shift_operator_name"] or "невідомо"
        decision_user = row["decision_user_name"] or "рішення ще немає"
        ppe_state = person_ppe_status_label(row["person_ppe_status"])
        return (
            f"#{row['id']} | {camera_label(row['camera_id'])} | {row['created_at']} | "
            f"рішення: {status_label(row['status'])} | висновок: {ppe_state} | "
            f"оцінка ризику {float(row['risk_score']):.0%} | {row['source_file']} | "
            f"оператор: {operator} | рішення прийняв: {decision_user}"
        )

    def details_text(self, row: sqlite3.Row) -> str:
        decision = "Рішення ще не прийнято"
        if row["last_decision_at"]:
            decision = (
                f"{row['last_decision_at']}: {status_label(row['last_previous_status'])} -> "
                f"{status_label(row['last_new_status'])}; користувач: {row['decision_user_name'] or 'невідомо'}"
            )
        track_id = row["track_id"] if row["track_id"] is not None else "немає"
        ppe_summary = row["ppe_summary"] or "немає повної перевірки ЗІЗ"
        ppe_status = person_ppe_status_label(row["person_ppe_status"])
        resolved_at = row["resolved_at"] or "не зафіксовано"
        resolved_frame = row["resolved_frame_index"] if row["resolved_frame_index"] is not None else "немає"
        resolved_summary = row["resolved_ppe_summary"] or "немає"
        resolved_path = row["resolved_screenshot_path"] or "немає"
        shift_comment = row["shift_comment"] or "немає коментаря"
        decided_at = row["decided_at"] or "ще не вирішено"
        decided_by = row["decision_user_name"] or "ще не вирішено"
        if row["person_ppe_status"] == "unclear":
            risk_explanation = (
                "Базовий пріоритет ручної перевірки: модель виявила людину, "
                "але не визначила каску або жилет. Це не підтвердження порушення."
            )
        elif row["person_ppe_status"] == "violation":
            risk_explanation = "Оцінка сформована з упевненості моделі в ознаці порушення."
        else:
            risk_explanation = "Після подальшого спостереження модель зафіксувала наявність ЗІЗ."
        return "\n".join(
            [
                f"Інцидент: #{row['id']} / Унікальний код: {row['uid']}",
                f"Рішення оператора: {status_label(row['status'])}",
                f"Створено: {row['created_at']}",
                f"Рішення: {decided_at}; хто прийняв: {decided_by}",
                f"Остання зміна рішення: {decision}",
                "",
                f"Камера: {camera_label(row['camera_id'])}",
                f"Джерело: {row['source_file']}",
                f"Кадр відео: {row['frame_index']}",
                f"Висновок моделі для картки: {detection_label(str(row['label']))}",
                f"Перевірка ЗІЗ: {ppe_summary}",
                f"Стан ЗІЗ за даними моделі: {ppe_status}",
                f"Коли зафіксовано виправлення: {resolved_at}",
                f"Кадр виправлення: {resolved_frame}",
                f"Перевірка ЗІЗ на кадрі виправлення: {resolved_summary}",
                f"Оцінка ризику / пріоритет перевірки: {float(row['risk_score']):.0%}",
                f"Пояснення оцінки: {risk_explanation}",
                f"Впевненість розпізнавання людини: {format_confidence(row['person_confidence']) if row['person_confidence'] is not None else 'не виявлено'}",
                f"Впевненість розпізнавання каски: {format_confidence(row['helmet_confidence']) if row['helmet_confidence'] is not None else 'не виявлено'}",
                f"Впевненість розпізнавання жилета: {format_confidence(row['vest_confidence']) if row['vest_confidence'] is not None else 'не виявлено'}",
                f"Впевненість у відсутності каски: {format_confidence(row['no_helmet_confidence']) if row['no_helmet_confidence'] is not None else 'не виявлено'}",
                f"Впевненість у відсутності жилета: {format_confidence(row['no_vest_confidence']) if row['no_vest_confidence'] is not None else 'не виявлено'}",
                f"Номер людини у відстеженні: {track_id}",
                f"Службовий ключ порушення: {row['violation_key'] or 'немає'}",
                "",
                f"Зміна: #{row['shift_id'] or 'немає'}",
                f"Оператор зміни: {row['shift_operator_name'] or 'невідомо'}",
                f"Початок зміни: {row['shift_started_at'] or 'невідомо'}",
                f"Кінець зміни: {row['shift_ended_at'] or 'ще триває / не записано'}",
                f"Коментар зміни: {shift_comment}",
                "",
                f"Поточний файл кадру: {row['current_file_path']}",
                f"Кадр виправлення: {resolved_path}",
                f"Початковий кадр без позначок: {row['raw_path'] or 'немає'}",
                f"Копія для донавчання моделі: {row['active_learning_path'] or 'немає'}",
            ]
        )

    def open_selected_folder(self) -> None:
        row = self.current_row()
        if row is None:
            return
        path = absolute_path(self.display_frame_path(row))
        folder = path.parent if path and path.parent.exists() else DATA
        self.log_action("open_frame_folder", f"Відкрито папку кадру інциденту #{row['id']}: {folder}")
        os.startfile(folder)


class AdminPanel(QWidget):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store

        layout = QVBoxLayout(self)
        title = QLabel("Користувачі системи")
        title.setObjectName("title")
        layout.addWidget(title)
        form = QGridLayout()
        self.login = QLineEdit()
        self.name = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.operator = QCheckBox("Оператор")
        self.operator.setChecked(True)
        self.admin = QCheckBox("Адмін")
        form.addWidget(QLabel("Логін"), 0, 0)
        form.addWidget(self.login, 0, 1)
        form.addWidget(QLabel("Ім'я"), 1, 0)
        form.addWidget(self.name, 1, 1)
        form.addWidget(QLabel("Новий пароль"), 2, 0)
        form.addWidget(self.password, 2, 1)
        form.addWidget(self.operator, 3, 0)
        form.addWidget(self.admin, 3, 1)
        layout.addLayout(form)

        save_btn = QPushButton("Створити / зберегти користувача")
        save_btn.clicked.connect(self.save_user)
        layout.addWidget(save_btn)

        self.users = QListWidget()
        self.users.itemClicked.connect(self.load_user_from_item)
        layout.addWidget(self.users)
        deactivate_btn = QPushButton("Деактивувати вибраного користувача")
        deactivate_btn.clicked.connect(self.deactivate_selected)
        layout.addWidget(deactivate_btn)
        self.refresh()

    def refresh(self) -> None:
        self.users.clear()
        for row in self.store.list_users():
            roles = []
            if row["is_operator"]:
                roles.append("оператор")
            if row["is_admin"]:
                roles.append("адмін")
            active = "активний" if row["active"] else "вимкнений"
            item_text = f"{row['id']} | {row['display_name']} | {row['login']} | {', '.join(roles)} | {active}"
            self.users.addItem(item_text)

    def load_user_from_item(self, item) -> None:
        user_id = int(item.text().split("|", 1)[0].strip())
        row = self.store.fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
        if row:
            self.login.setText(row["login"])
            self.name.setText(row["display_name"])
            self.password.clear()
            self.operator.setChecked(bool(row["is_operator"]))
            self.admin.setChecked(bool(row["is_admin"]))

    def save_user(self) -> None:
        message = self.store.save_user(self.login.text(), self.name.text(), self.password.text(), self.operator.isChecked(), self.admin.isChecked())
        if message.startswith(("Не зареєстровано", "Не збережено")):
            show_warning(self, "Користувачі", message)
            return
        show_information(self, "Користувачі", message)
        self.refresh()

    def deactivate_selected(self) -> None:
        item = self.users.currentItem()
        if not item:
            show_warning(self, "Користувачі", "Оберіть користувача для деактивації.")
            return
        user_id = int(item.text().split("|", 1)[0].strip())
        confirmed = ask_yes_no(
            self,
            "Деактивувати користувача?",
            "Користувач не буде видалений з БД, але не зможе увійти в систему.",
        )
        if not confirmed:
            return
        message = self.store.deactivate_user(user_id)
        if message.startswith("Не деактивовано"):
            show_warning(self, "Користувачі", message)
        else:
            show_information(self, "Користувачі", message)
        self.refresh()


class MainWindow(QMainWindow):
    shift_ended = Signal()

    def __init__(self, store: Store, detector: Detector, user: User):
        super().__init__()
        self.store = store
        self.user = user
        self.shift_id = self.store.start_shift(user)
        self.box_mode = "violations"
        self.detectors = {"CAM_01": detector, "CAM_02": Detector()}
        self.camera_model_status = {camera_id: "завантажується..." for camera_id in CAMERA_IDS}
        self.current_image_paths: dict[str, str | None] = {camera_id: None for camera_id in CAMERA_IDS}
        self.worker_threads: dict[str, QThread | None] = {camera_id: None for camera_id in CAMERA_IDS}
        self.workers: dict[str, VideoWorker | None] = {camera_id: None for camera_id in CAMERA_IDS}
        self.video_labels: dict[str, QLabel] = {}
        self.camera_status_labels: dict[str, QLabel] = {}
        self.next_camera_index = 0
        self.shift_closed = False
        self.return_to_login_after_close = False
        self.admin_panel: AdminPanel | None = None

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1200, 760)
        self.build_ui()
        self.setup_timers()
        self.refresh_all()

        for camera_id, camera_detector in self.detectors.items():
            camera_detector.loaded.connect(
                lambda message, ok, camera_id=camera_id: self.on_model_loaded(camera_id, message, ok)
            )
            threading.Thread(target=camera_detector.load, daemon=True).start()

    def build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        title = QLabel(APP_TITLE)
        title.setObjectName("headerTitle")
        self.clock_label = QLabel("")
        end_btn = QPushButton("Завершити зміну")
        end_btn.clicked.connect(self.ask_end_shift)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel(f"Користувач: {self.user.display_name}"))
        header_layout.addWidget(self.clock_label)
        header_layout.addWidget(end_btn)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        root.addLayout(body, 1)

        left = QFrame()
        left.setObjectName("leftPanel")
        left.setFixedWidth(LEFT_COLUMN_WIDTH)
        left_layout = QVBoxLayout(left)
        top_row = QHBoxLayout()
        open_btn = QPushButton("Відкрити фото/відеофайл")
        open_btn.clicked.connect(self.pick_files)
        top_row.addWidget(open_btn)
        self.assignment_label = QLabel(f"Наступне джерело: {camera_label(CAMERA_IDS[self.next_camera_index])}")
        top_row.addWidget(self.assignment_label, 1)
        left_layout.addLayout(top_row)

        self.source_label = QLabel("Стан: очікування джерел")
        left_layout.addWidget(self.source_label)
        for camera_id in CAMERA_IDS:
            camera_panel = QFrame()
            camera_panel.setObjectName("panel")
            camera_layout = QVBoxLayout(camera_panel)
            title_row = QHBoxLayout()
            title_row.addWidget(QLabel(camera_label(camera_id)))
            status_label = QLabel("Джерело не підключено")
            title_row.addWidget(status_label, 1)
            camera_layout.addLayout(title_row)
            video_label = QLabel("Підключіть відео або фото")
            video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_label.setObjectName("video")
            video_label.setMinimumHeight(VIDEO_VIEW_MIN_HEIGHT)
            camera_layout.addWidget(video_label, 1)
            self.camera_status_labels[camera_id] = status_label
            self.video_labels[camera_id] = video_label
            left_layout.addWidget(camera_panel, 1)
        body.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.monitor_tab(), "Моніторити")
        self.tabs.addTab(self.archive_tab(), "Сортувати з архіву")
        self.tabs.addTab(self.service_tab(), "Сервіс")
        self.tabs.currentChanged.connect(self.on_main_tab_changed)
        body.addWidget(self.tabs, 1)

    def setup_timers(self) -> None:
        timer = QTimer(self)
        timer.timeout.connect(self.update_clock)
        timer.start(1000)
        self.update_clock()

    def update_clock(self) -> None:
        self.clock_label.setText(dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

    def monitor_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.model_label = QLabel("Модель: завантажується...")
        layout.addWidget(self.model_label)
        layout.addLayout(self.box_mode_controls())

        self.monitor_panel = IncidentPanel("Останнє порушення в черзі", self.apply_decision, self.undo_last, self.refresh_cards, "latest")
        layout.addWidget(self.monitor_panel, 1)
        return tab

    def archive_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.archive_panel = IncidentPanel("Найдавніша картка, що очікує рішення", self.apply_decision, self.undo_last, self.refresh_cards, "oldest")
        layout.addWidget(self.archive_panel, 1)
        return tab

    def box_mode_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        all_btn = QRadioButton("Усі рамки")
        violations_btn = QRadioButton("Лише порушення")
        none_btn = QRadioButton("Без рамок")
        violations_btn.setChecked(True)
        for btn, value in [(all_btn, "all"), (violations_btn, "violations"), (none_btn, "none")]:
            btn.toggled.connect(lambda checked, mode=value: self.change_box_mode(mode) if checked else None)
            row.addWidget(btn)
        return row

    def dashboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        stats = QWidget()
        self.dashboard_layout = QVBoxLayout(stats)
        layout.addWidget(stats)
        layout.addWidget(QLabel("Журнал подій"))
        self.events = QListWidget()
        layout.addWidget(self.events, 1)
        return tab

    def service_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        menu = QHBoxLayout()
        menu.setSpacing(6)
        search_btn = QPushButton("Пошук інцидентів")
        dashboard_btn = QPushButton("Дашборд")
        users_btn = QPushButton("Користувачі системи")
        self.service_page_buttons = [search_btn, dashboard_btn, users_btn]
        for button in self.service_page_buttons:
            button.setCheckable(True)
            menu.addWidget(button)
        menu.addStretch(1)
        layout.addLayout(menu)

        self.service_stack = QStackedWidget()
        self.incident_search = IncidentSearchPanel(self.store, self)
        self.service_stack.addWidget(self.incident_search)
        self.service_stack.addWidget(self.dashboard_tab())
        search_btn.clicked.connect(lambda: self.show_service_page(0))
        dashboard_btn.clicked.connect(lambda: self.show_service_page(1))
        users_btn.clicked.connect(self.ask_admin)

        layout.addWidget(self.service_stack, 1)

        notes = QFrame()
        notes.setObjectName("panel")
        notes_layout = QVBoxLayout(notes)
        notes_layout.addWidget(QLabel("Нотатки активної зміни"))
        self.shift_note_input = QTextEdit()
        self.shift_note_input.setFixedHeight(62)
        self.shift_note_input.setPlaceholderText("Занотувати щось про зміну, систему або інцидент...")
        notes_layout.addWidget(self.shift_note_input)
        notes_row = QHBoxLayout()
        add_note_btn = QPushButton("Додати нотатку")
        history_btn = QPushButton("Нотатки попередніх змін")
        add_note_btn.clicked.connect(self.add_shift_note)
        history_btn.clicked.connect(self.show_shift_notes)
        notes_row.addWidget(add_note_btn)
        notes_row.addWidget(history_btn)
        notes_row.addStretch(1)
        notes_layout.addLayout(notes_row)
        layout.addWidget(notes)
        self.show_service_page(0)
        return tab

    def show_service_page(self, index: int) -> None:
        if index >= self.service_stack.count():
            return
        self.service_stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.service_page_buttons):
            button.setChecked(button_index == index)
        if index == 0:
            self.incident_search.refresh_sources()
        elif index == 1:
            self.refresh_dashboard()
            self.refresh_service_log()
        elif index == 2 and self.admin_panel is not None:
            self.admin_panel.refresh()

    def on_main_tab_changed(self, index: int) -> None:
        if index != 2 and self.service_stack.currentIndex() == 2:
            self.show_service_page(0)

    def on_model_loaded(self, camera_id: str, message: str, ok: bool) -> None:
        self.camera_model_status[camera_id] = "готова" if ok else "помилка"
        status = " | ".join(
            f"{camera_label(item)}: {self.camera_model_status[item]}" for item in CAMERA_IDS
        )
        self.model_label.setText(f"Моделі: {status}")
        self.log_action(
            "model_loaded" if ok else "model_error",
            f"{camera_label(camera_id)}: {message}",
        )

    def log_action(self, event_type: str, message: str) -> None:
        self.store.log_event(self.shift_id, self.user.id, event_type, message)

    def camera_is_running(self, camera_id: str) -> bool:
        thread = self.worker_threads[camera_id]
        return bool(thread and thread.isRunning())

    def camera_for_new_source(self) -> str:
        preferred = CAMERA_IDS[self.next_camera_index]
        other = CAMERA_IDS[(self.next_camera_index + 1) % len(CAMERA_IDS)]
        if not self.camera_is_running(preferred):
            return preferred
        if not self.camera_is_running(other):
            return other
        return preferred

    def advance_camera_turn(self, assigned_camera_id: str) -> None:
        index = CAMERA_IDS.index(assigned_camera_id)
        self.next_camera_index = (index + 1) % len(CAMERA_IDS)
        self.assignment_label.setText(f"Наступне джерело: {camera_label(CAMERA_IDS[self.next_camera_index])}")

    def set_camera_status(self, camera_id: str, message: str) -> None:
        self.camera_status_labels[camera_id].setText(message)
        self.camera_status_labels[camera_id].setToolTip(message)

    def pick_files(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Оберіть відео або фото",
            str(ROOT),
            "Медіафайли (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if not path:
            self.log_action("file_dialog_cancel", "Оператор закрив вибір файлу без вибору джерела.")
            return
        camera_id = self.camera_for_new_source()
        detector = self.detectors[camera_id]
        if not detector.ready:
            show_warning(self, "Модель", f"Модель для {camera_label(camera_id)} ще завантажується. Зачекайте кілька секунд.")
            self.log_action("source_blocked", f"{camera_label(camera_id)}: спроба відкрити файл до завершення завантаження моделі.")
            return
        if self.camera_is_running(camera_id):
            replace = ask_yes_no(
                self,
                "Замінити відео?",
                f"{camera_label(camera_id)} ще аналізує відео. Зупинити його та підключити нове джерело?",
            )
            if not replace:
                return
            if not self.stop_camera_worker(camera_id):
                show_warning(self, "Відео", "Попереднє відео ще завершує роботу. Спробуйте ще раз за кілька секунд.")
                return
        self.source_label.setText(f"Стан: підключено джерело до {camera_label(camera_id)}")
        self.log_action("file_selected", f"{camera_label(camera_id)}: обрано файл для аналізу: {Path(path).name}")
        self.process_source(path, camera_id)
        self.advance_camera_turn(camera_id)

    def process_source(self, path: str, camera_id: str) -> None:
        ext = Path(path).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self.current_image_paths[camera_id] = path
            self.process_image(path, camera_id, create_card=True)
        else:
            self.current_image_paths[camera_id] = None
            self.start_video(path, camera_id)

    def process_image(self, path: str, camera_id: str, create_card: bool) -> None:
        frame = cv2.imread(path)
        if frame is None:
            show_warning(self, "Фото", f"Не вдалося відкрити фото: {Path(path).name}")
            self.log_action("source_error", f"{camera_label(camera_id)}: не вдалося відкрити фото: {Path(path).name}")
            return
        self.log_action("image_analysis_start", f"{camera_label(camera_id)}: почато аналіз фото: {Path(path).name}")
        draw, detections = self.detectors[camera_id].analyze(frame, self.box_mode)
        self.set_video_pixmap(camera_id, jpg_to_pixmap(frame_to_jpg(draw)))
        self.set_camera_status(camera_id, f"{Path(path).name} | фото проаналізовано")
        self.log_action("image_analysis_done", f"{camera_label(camera_id)}: фото проаналізовано: {Path(path).name}, виявлень={len(detections)}")
        if detections and create_card:
            created = self.create_image_incident(camera_id, path, frame, draw, detections)
            self.set_camera_status(camera_id, f"{Path(path).name} | фото, створено карток: {created}")
            self.refresh_all()

    def create_image_incident(self, camera_id: str, path: str, raw_frame, draw_frame, detections: list[dict[str, Any]]) -> int:
        created = 0
        for group in Detector.group_person_checks(detections):
            if not bool(group.get("needs_review", True)) or group.get("person_ppe_status") == "ppe_ok":
                continue
            group_detections = group["detections"]
            labels = sorted({str(item["label"]) for item in group_detections})
            label = "+".join(labels)
            risk_score = float(group["risk_score"])
            ppe_summary = str(group.get("ppe_summary") or "")
            track_id = int(group["track_id"]) if group.get("track_id") is not None else None
            track_part = f"_P{track_id}" if track_id is not None else f"_G{created + 1}"
            name = f"{camera_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{track_part}_{label}.jpg"
            screenshot = FOLDERS["fixed_new"] / name
            raw = FOLDERS["raw"] / name
            incident_frame = raw_frame.copy()
            Detector.draw_person_checks(incident_frame, group_detections, show_annotations=False)
            try:
                write_frame_atomic(screenshot, incident_frame)
                write_frame_atomic(raw, raw_frame)
                incident_id = self.store.add_incident(
                    camera_id=camera_id,
                    source_file=Path(path).name,
                    frame_index=1,
                    label=label,
                    person_ppe_status=str(group.get("person_ppe_status") or "violation"),
                    risk_score=risk_score,
                    original_screenshot_path=str(screenshot),
                    raw_path=str(raw),
                    shift_id=self.shift_id,
                    track_id=track_id,
                    violation_key=f"{camera_id}:{Path(path).name}:{group['key']}",
                    ppe_summary=ppe_summary,
                    person_confidence=group.get("person_confidence"),
                    helmet_confidence=group.get("helmet_confidence"),
                    vest_confidence=group.get("vest_confidence"),
                    no_helmet_confidence=group.get("no_helmet_confidence"),
                    no_vest_confidence=group.get("no_vest_confidence"),
                )
            except Exception:
                remove_file_if_exists(screenshot)
                remove_file_if_exists(raw)
                raise
            created += 1
            self.log_action(
                "alert",
                f"Створено картку перевірки #{incident_id}: камера={camera_label(camera_id)}, джерело={Path(path).name}, висновок моделі={detection_label(label)}, пріоритет перевірки={risk_score:.4f}, група={group['key']}, ЗІЗ={ppe_summary}",
            )
        return created

    def start_video(self, source: str, camera_id: str) -> None:
        if not self.stop_camera_worker(camera_id):
            show_warning(self, "Відео", f"{camera_label(camera_id)} ще завершує попередній аналіз. Нове відео не запущено.")
            self.log_action("source_blocked", f"{camera_label(camera_id)}: нове джерело не запущено, попередній аналіз ще завершується: {source}")
            return
        label = Path(source).name
        self.set_camera_status(camera_id, f"{label} | аналіз відео")
        thread = QThread()
        self.log_action("source_start_request", f"{camera_label(camera_id)}: запит запуску відеофайлу: {label}")
        worker = VideoWorker(camera_id, source, self.detectors[camera_id], self.store, self.shift_id, self.user.id, lambda: self.box_mode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_ready.connect(lambda data, camera_id=camera_id: self.set_video_pixmap(camera_id, jpg_to_pixmap(data)))
        worker.status.connect(lambda message, camera_id=camera_id: self.set_camera_status(camera_id, message))
        worker.incident_created.connect(self.refresh_all)
        worker.finished.connect(lambda camera_id=camera_id, worker=worker: self.on_worker_finished(camera_id, worker))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda camera_id=camera_id, thread=thread: self.on_thread_finished(camera_id, thread))
        thread.finished.connect(thread.deleteLater)
        self.worker_threads[camera_id] = thread
        self.workers[camera_id] = worker
        thread.start()

    def on_worker_finished(self, camera_id: str, worker: VideoWorker) -> None:
        if self.workers[camera_id] is worker:
            self.workers[camera_id] = None
        self.refresh_service_log()

    def on_thread_finished(self, camera_id: str, thread: QThread) -> None:
        if self.worker_threads[camera_id] is thread:
            self.worker_threads[camera_id] = None

    def stop_camera_worker(self, camera_id: str) -> bool:
        worker = self.workers[camera_id]
        thread = self.worker_threads[camera_id]
        if worker:
            try:
                worker.stop()
            except RuntimeError:
                self.workers[camera_id] = None
                worker = None
        if thread and thread.isRunning():
            self.set_camera_status(camera_id, "Попереднє відео зупиняється...")
            QApplication.processEvents()
            thread.quit()
            if not thread.wait(WORKER_STOP_TIMEOUT_MS):
                self.log_action("source_stop_error", f"{camera_label(camera_id)}: не вдалося швидко зупинити відеоаналіз.")
                return False
            self.log_action("source_stop_request", f"{camera_label(camera_id)}: попередній відеоаналіз зупинено.")
        elif thread:
            try:
                thread.wait(100)
            except RuntimeError:
                pass
        self.workers[camera_id] = None
        self.worker_threads[camera_id] = None
        return True

    def stop_all_workers(self) -> bool:
        stopped = True
        for camera_id in CAMERA_IDS:
            stopped = self.stop_camera_worker(camera_id) and stopped
        return stopped

    def set_video_pixmap(self, camera_id: str, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        video_label = self.video_labels[camera_id]
        video_label.setPixmap(
            pixmap.scaled(
                video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for camera_id, video_label in self.video_labels.items():
            if video_label.pixmap():
                self.set_video_pixmap(camera_id, video_label.pixmap())
        self.refresh_cards()

    def change_box_mode(self, mode: str) -> None:
        self.box_mode = mode
        visible_mode = {
            "all": "усі позначки",
            "violations": "лише порушення",
            "none": "без позначок",
        }.get(mode, mode)
        self.log_action("view_mode", f"Режим рамок змінено: {visible_mode}")
        for camera_id, image_path in self.current_image_paths.items():
            if image_path:
                self.process_image(image_path, camera_id, create_card=False)

    def apply_decision(self, order: str, status: str) -> None:
        panel = self.monitor_panel if order == "latest" else self.archive_panel
        row = None
        if panel.current_id is not None:
            row = self.store.fetchone("SELECT * FROM incidents WHERE id = ? AND status = 'pending'", (panel.current_id,))
        if row is None:
            row = self.store.pending_incident(order)
        if row is None:
            self.source_label.setText("Немає картки для рішення.")
            self.log_action("decision_empty", f"Оператор обрав рішення «{status_label(status)}», але картки, що очікує рішення, немає.")
            return
        message = self.store.decide(row["id"], status, self.user.id, self.shift_id)
        self.source_label.setText(message)
        self.refresh_all()

    def undo_last(self) -> None:
        self.source_label.setText(self.store.undo_last_decision(self.user.id, self.shift_id))
        self.refresh_all()

    def refresh_all(self) -> None:
        self.refresh_cards()
        self.refresh_dashboard()
        self.refresh_service_log()

    def refresh_cards(self) -> None:
        self.monitor_panel.set_incident(self.store.pending_incident("latest"))
        self.archive_panel.set_incident(self.store.pending_incident("oldest"))

    def refresh_dashboard(self) -> None:
        while self.dashboard_layout.count():
            item = self.dashboard_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        stats = self.store.stats()
        rows = [
            ("Зафіксовано системою", stats["total"]),
            ("Підтверджено людиною", stats["confirmed"]),
            ("Хибне спрацювання", stats["false_positive"]),
            ("Невизначено людиною", stats["ignored"]),
            ("Очікує рішення", stats["pending"]),
        ]
        max_value = max(stats["total"], 1)
        for label, value in rows:
            self.dashboard_layout.addWidget(QLabel(f"{label}: {value}"))
            bar = QProgressBar()
            bar.setRange(0, max_value)
            bar.setValue(value)
            self.dashboard_layout.addWidget(bar)

        self.dashboard_layout.addWidget(
            QLabel(
                f"Частка хибних спрацювань серед переглянутих: {stats['false_positive_rate']:.0%} "
                f"({stats['false_positive']} з {stats['reviewed']})"
            )
        )
        false_positive_rate = QProgressBar()
        false_positive_rate.setRange(0, 100)
        false_positive_rate.setValue(int(stats["false_positive_rate"] * 100))
        self.dashboard_layout.addWidget(false_positive_rate)
        self.dashboard_layout.addWidget(QLabel(f"Середня оцінка ризику всіх карток: {stats['avg_risk']:.0%}"))
        all_risk = QProgressBar()
        all_risk.setRange(0, 100)
        all_risk.setValue(int(stats["avg_risk"] * 100))
        self.dashboard_layout.addWidget(all_risk)
        self.dashboard_layout.addWidget(QLabel(f"Середня оцінка ризику підтверджених інцидентів: {stats['confirmed_avg_risk']:.0%}"))
        confirmed_risk = QProgressBar()
        confirmed_risk.setRange(0, 100)
        confirmed_risk.setValue(int(stats["confirmed_avg_risk"] * 100))
        self.dashboard_layout.addWidget(confirmed_risk)
        self.dashboard_layout.addStretch(1)

    def refresh_service_log(self) -> None:
        self.events.clear()
        for row in self.store.recent_events():
            event_type = event_type_label(str(row["event_type"]))
            message = display_event_message(str(row["message"]))
            self.events.addItem(f"{row['created_at']} | {event_type} | {message}")

    def add_shift_note(self) -> None:
        note = self.shift_note_input.toPlainText().strip()
        if not note:
            self.source_label.setText("Нотатку не додано: введіть текст.")
            return
        if not self.store.add_shift_note(self.shift_id, self.user, note):
            self.source_label.setText("Нотатку не додано: активну зміну не знайдено.")
            return
        self.shift_note_input.clear()
        self.source_label.setText("Нотатку додано до активної зміни.")
        self.refresh_service_log()

    def show_shift_notes(self) -> None:
        rows = self.store.recent_shift_notes()
        text = "\n\n".join(
            f"Зміна #{row['id']} | {row['display_name']} | {row['started_at']}\n{row['comment']}"
            for row in rows
        ) or "Нотаток у змінах ще немає."
        dialog = QDialog(self)
        dialog.setWindowTitle("Нотатки змін")
        dialog.resize(600, 430)
        layout = QVBoxLayout(dialog)
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(text)
        layout.addWidget(notes)
        close_btn = QPushButton("Закрити")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    def ask_admin(self) -> None:
        self.log_action("admin_open_request", "Оператор відкрив запит адміністративного доступу.")
        password, ok = PasswordDialog.ask(self)
        if not ok:
            self.log_action("admin_open_cancel", "Адміністративний доступ скасовано.")
            self.show_service_page(self.service_stack.currentIndex())
            return
        if not self.store.verify_admin_password(password):
            show_warning(self, "Адмін", "Невірний пароль адміністратора.")
            self.log_action("admin_open_denied", "Введено неправильний пароль адміністратора.")
            self.show_service_page(self.service_stack.currentIndex())
            return
        if self.admin_panel is None:
            self.admin_panel = AdminPanel(self.store)
            self.service_stack.addWidget(self.admin_panel)
        self.log_action("admin_open_success", "Відкрито панель керування користувачами.")
        self.show_service_page(2)

    def ask_end_shift(self) -> None:
        confirmed = ask_yes_no(
            self,
            "ВИ ХОЧЕТЕ ЗАВЕРШИТИ СЕАНС?",
            "Поточна зміна буде завершена, після цього відкриється вікно входу для наступного користувача.",
        )
        if confirmed:
            self.log_action("shift_finish_confirm", "Оператор підтвердив завершення зміни.")
            self.finish_shift()
        else:
            self.log_action("shift_finish_cancel", "Оператор скасував завершення зміни.")

    def finish_shift(self) -> bool:
        if not self.prepare_window_close("shift_finish"):
            return False
        self.log_action("shift_finish_start", "Почато завершення зміни.")
        self.end_current_shift()
        self.return_to_login_after_close = True
        if not self.close():
            self.return_to_login_after_close = False
            self.log_action("shift_finish_blocked", "Зміну не завершено: вікно не закрилось.")
            return False
        QTimer.singleShot(0, lambda: self.shift_ended.emit())
        return True

    def end_current_shift(self) -> None:
        if self.shift_closed:
            return
        self.shift_closed = self.store.end_shift(self.shift_id, self.user.id)

    def prepare_window_close(self, reason: str) -> bool:
        if not self.stop_all_workers():
            message = "Закриття вікна заблоковано: відеоаналіз ще завершується."
            if reason == "shift_finish":
                message = "Зміну не завершено: відеоаналіз не зупинився вчасно."
                show_warning(self, "Відео", "Не вдалося швидко зупинити відео. Спробуйте завершити зміну ще раз.")
            self.log_action(f"{reason}_blocked", message)
            return False
        return True

    def close_directly(self) -> None:
        self.log_action("window_close", "Вікно закрито напряму; зміна завершується автоматично.")
        self.end_current_shift()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)

    def closeEvent(self, event) -> None:
        if not self.prepare_window_close("window_close"):
            event.ignore()
            return
        if not self.return_to_login_after_close:
            self.close_directly()
        event.accept()


class PasswordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Адміністраторський доступ")
        layout = QVBoxLayout(self)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("Пароль адміністратора")
        layout.addWidget(self.password)
        row = QHBoxLayout()
        cancel = QPushButton("Скасувати")
        enter = QPushButton("Увійти")
        cancel.clicked.connect(self.reject)
        enter.clicked.connect(self.accept)
        enter.setDefault(True)
        self.password.returnPressed.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(enter)
        layout.addLayout(row)

    @staticmethod
    def ask(parent=None) -> tuple[str, bool]:
        dialog = PasswordDialog(parent)
        ok = dialog.exec() == QDialog.DialogCode.Accepted
        return dialog.password.text(), ok


def apply_style(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget {
            background: #080B0E;
            color: #E8EEF4;
            font-size: 14px;
        }
        QFrame#header, QFrame#panel, QTabWidget::pane {
            background: #14191F;
            border: 1px solid #34404C;
        }
        QTabBar::tab {
            background: #10151B;
            color: #E8EEF4;
            border: 1px solid #34404C;
            padding: 8px 14px;
            min-width: 120px;
        }
        QTabBar::tab:selected {
            background: #26313C;
            color: #FFFFFF;
            border-bottom-color: #26313C;
        }
        QTabBar::tab:hover {
            background: #1B222A;
        }
        QMenuBar, QMenuBar::item, QMenu {
            background: #14191F;
            color: #E8EEF4;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background: #26313C;
            color: #FFFFFF;
        }
        QLabel#headerTitle, QLabel#title {
            font-size: 22px;
            font-weight: 700;
        }
        QLabel#error {
            color: #EB5757;
        }
        QLabel#video, QLabel#incidentImage {
            background: #050607;
            border: 1px solid #34404C;
            border-radius: 6px;
        }
        QPushButton {
            background: #1B222A;
            border: 1px solid #34404C;
            padding: 8px 12px;
            border-radius: 6px;
        }
        QPushButton:hover {
            background: #26313C;
        }
        QPushButton:checked {
            background: #2F80ED;
            color: #FFFFFF;
            border-color: #2F80ED;
        }
        QLineEdit, QDateEdit, QComboBox, QTextEdit, QListWidget {
            background: #10151B;
            border: 1px solid #34404C;
            padding: 6px;
            border-radius: 5px;
        }
        QProgressBar {
            border: 1px solid #34404C;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2F80ED;
        }
        """
    )


class SessionController(QObject):
    def __init__(self, app: QApplication, store: Store):
        super().__init__()
        self.app = app
        self.store = store
        self.window: MainWindow | None = None

    def current_context(self) -> tuple[int | None, int | None]:
        if self.window is None:
            return None, None
        return self.window.shift_id, self.window.user.id

    def log_app_event(self, event_type: str, message: str) -> None:
        shift_id, user_id = self.current_context()
        self.store.log_event(shift_id, user_id, event_type, message)

    def log_app_exit(self) -> None:
        try:
            self.log_app_event("app_exit", "Qt повідомив про завершення роботи застосунку.")
        except Exception:
            pass

    def start(self) -> None:
        QTimer.singleShot(0, self.show_login)

    def show_login(self) -> None:
        login = LoginDialog(self.store)
        if login.exec() != QDialog.DialogCode.Accepted or login.user is None:
            self.app.quit()
            return
        self.window = MainWindow(self.store, Detector(), login.user)
        self.window.shift_ended.connect(self.start_next_shift)
        self.window.showFullScreen()

    def start_next_shift(self) -> None:
        if self.window is not None:
            self.window.deleteLater()
        self.window = None
        QTimer.singleShot(0, self.show_login)


