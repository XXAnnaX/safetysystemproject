from __future__ import annotations

import sys
import threading
import traceback

from PySide6.QtWidgets import QApplication

from config import DB_PATH, ensure_storage
from store import Store
from ui import SessionController, apply_style


def main() -> int:
    ensure_storage()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    apply_style(app)
    store = Store(DB_PATH)
    controller = SessionController(app, store)
    restored_files, isolated_files = store.recover_file_consistency()
    recovered = store.recover_open_shifts()
    if restored_files or isolated_files:
        store.log_event(
            None,
            None,
            "app_recovery",
            f"Перевірено файли кадрів: відновлено шляхів {restored_files}, ізольовано зайвих файлів {isolated_files}.",
        )
    if recovered:
        store.log_event(None, None, "app_start", f"Додаток запущено. Автоматично закрито незавершених змін: {recovered}.")
    else:
        store.log_event(None, None, "app_start", "Додаток запущено.")

    previous_excepthook = sys.excepthook
    previous_thread_excepthook = threading.excepthook

    def log_uncaught_exception(exc_type, exc_value, exc_tb) -> None:
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-4000:]
        try:
            controller.log_app_event("app_exception", message)
        except Exception:
            pass
        previous_excepthook(exc_type, exc_value, exc_tb)

    def log_thread_exception(args) -> None:
        message = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))[-4000:]
        try:
            controller.log_app_event("app_thread_exception", message)
        except Exception:
            pass
        previous_thread_excepthook(args)

    sys.excepthook = log_uncaught_exception
    threading.excepthook = log_thread_exception
    app.aboutToQuit.connect(controller.log_app_exit)
    controller.start()
    result = app.exec()
    try:
        controller.log_app_event("app_exec_finished", f"Цикл застосунку завершено з кодом {result}.")
    except Exception:
        pass
    return result


if __name__ == "__main__":
    raise SystemExit(main())
