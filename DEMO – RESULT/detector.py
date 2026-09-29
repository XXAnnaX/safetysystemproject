from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, Signal
from ultralytics import YOLO

from config import (
    CONFIDENCE,
    CURRENT_MODEL,
    DISPLAY_JPEG_QUALITY,
    FILE_PROCESS_EVERY_N,
    FOLDERS,
    HELMET_CLASS,
    MODEL_IMAGE_SIZE,
    MODEL_INFERENCE_LOCK,
    MODEL_SLOW_WARNING_SEC,
    NO_HELMET_CLASS,
    NO_VEST_CLASS,
    PERSON_CLASS,
    PERSON_REVIEW_COOLDOWN_SEC,
    SAFE_CONFLICT_MARGIN,
    TRACKER_CONFIG,
    VEST_CLASS,
    VIDEO_MAX_WIDTH,
    camera_label,
    detection_label,
    frame_to_jpg,
    remove_file_if_exists,
    write_frame_atomic,
)
from store import Store


class Detector(QObject):
    loaded = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.model: YOLO | None = None
        self.model_name = "не завантажено"
        self.ready = False
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            if not CURRENT_MODEL.exists():
                raise FileNotFoundError(f"Не знайдено модель: {CURRENT_MODEL.name}")
            model = YOLO(str(CURRENT_MODEL))
            with self.lock:
                self.model = model
                self.model_name = CURRENT_MODEL.name
                self.ready = True
            self.loaded.emit(f"Модель: {CURRENT_MODEL.name} (готова)", True)
        except Exception as ex:
            self.loaded.emit(f"Модель: помилка - {ex}", False)

    def analyze(self, frame, box_mode: str) -> tuple[Any, list[dict[str, Any]]]:
        with self.lock:
            model = self.model
        if model is None:
            return frame, []

        with MODEL_INFERENCE_LOCK:
            results = model.predict(frame, conf=CONFIDENCE, imgsz=MODEL_IMAGE_SIZE, verbose=False)[0]
        return self.parse_results(frame, results, box_mode, use_tracking=False)

    def analyze_video(self, frame, box_mode: str) -> tuple[Any, list[dict[str, Any]]]:
        with self.lock:
            model = self.model
        if model is None:
            return frame, []

        with MODEL_INFERENCE_LOCK:
            results = model.track(frame, conf=CONFIDENCE, imgsz=MODEL_IMAGE_SIZE, verbose=False, persist=True, tracker=TRACKER_CONFIG)[0]
        return self.parse_results(frame, results, box_mode, use_tracking=True)

    def reset_tracking(self) -> None:
        with self.lock:
            model = self.model
        predictor = getattr(model, "predictor", None) if model is not None else None
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def parse_results(self, frame, results, box_mode: str, use_tracking: bool) -> tuple[Any, list[dict[str, Any]]]:
        draw = frame.copy()
        detections: list[dict[str, Any]] = []
        if not results.boxes:
            return draw, detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        ids = results.boxes.id.cpu().numpy() if use_tracking and results.boxes.id is not None else [None] * len(boxes)
        items: list[dict[str, Any]] = []
        persons: list[dict[str, Any]] = []

        for box, cls_id, conf, track_id in zip(boxes, classes, confs, ids):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id) if track_id is not None else None
            is_violation = cls_id in (NO_HELMET_CLASS, NO_VEST_CLASS)
            is_safe = cls_id in (HELMET_CLASS, VEST_CLASS)
            is_person = cls_id == PERSON_CLASS
            if not (is_violation or is_safe or is_person):
                continue

            if cls_id == HELMET_CLASS:
                label = "HELMET"
                color = (0, 180, 0)
            elif cls_id == VEST_CLASS:
                label = "VEST"
                color = (0, 180, 0)
            elif cls_id == NO_HELMET_CLASS:
                label = "NO_HELMET"
                color = (0, 0, 255)
            elif cls_id == NO_VEST_CLASS:
                label = "NO_VEST"
                color = (0, 0, 255)
            else:
                label = "PERSON"
                color = (0, 180, 0)

            item = {
                "label": label,
                "confidence": float(conf),
                "box": (x1, y1, x2, y2),
                "track_id": track_id,
                "object_track_id": track_id,
                "is_violation": is_violation,
                "is_safe": is_safe,
                "is_person": is_person,
                "color": color,
            }
            items.append(item)
            if is_person:
                persons.append(item)

        for item in items:
            if not item["is_person"]:
                person = self.match_person(item["box"], persons)
                if person and person.get("track_id") is not None:
                    item["track_id"] = person["track_id"]
                    item["person_box"] = person["box"]
                elif person:
                    item["person_box"] = person["box"]

        detections = self.build_person_checks(items, persons)

        if box_mode != "none":
            for item in items:
                should_draw = box_mode == "all" or (box_mode == "violations" and item["is_violation"])
                if not should_draw:
                    continue
                x1, y1, x2, y2 = item["box"]
                color = item["color"]
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    draw,
                    f"{item['confidence']:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
        return draw, detections

    @staticmethod
    def person_group_key(item: dict[str, Any], index: int) -> Any:
        if item.get("person_box") is not None:
            return ("person_box", item["person_box"])
        if item.get("track_id") is not None:
            return ("track", item["track_id"])
        return ("unmatched", index)

    @staticmethod
    def detection_group_key(detection: dict[str, Any], index: int) -> Any:
        if detection.get("track_id") is not None:
            return ("track", int(detection["track_id"]))
        if detection.get("person_box") is not None:
            return ("person_box", tuple(detection["person_box"]))
        return ("box", tuple(detection["box"]))

    @staticmethod
    def group_person_checks(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[Any, dict[str, Any]] = {}
        for index, detection in enumerate(detections):
            key = Detector.detection_group_key(detection, index)
            group = groups.setdefault(
                key,
                {
                    "key": key,
                    "detections": [],
                    "labels": set(),
                    "confidence": 0.0,
                    "risk_score": 0.0,
                    "track_id": detection.get("track_id"),
                    "box": detection.get("person_box") or detection["box"],
                    "ppe_summary": detection.get("ppe_summary", ""),
                    "person_ppe_status": detection.get("person_ppe_status", "violation"),
                    "needs_review": bool(detection.get("needs_review", True)),
                    "person_confidence": detection.get("person_confidence"),
                    "helmet_confidence": detection.get("helmet_confidence"),
                    "vest_confidence": detection.get("vest_confidence"),
                    "no_helmet_confidence": detection.get("no_helmet_confidence"),
                    "no_vest_confidence": detection.get("no_vest_confidence"),
                },
            )
            group["detections"].append(detection)
            group["labels"].add(str(detection["label"]))
            group["confidence"] = max(group["confidence"], float(detection["confidence"]))
            group["risk_score"] = max(group["risk_score"], float(detection.get("risk_score", 0.0)))
            group["box"] = Detector.union_boxes(group["box"], detection.get("person_box") or detection["box"])
            if group["track_id"] is None and detection.get("track_id") is not None:
                group["track_id"] = detection["track_id"]
            if not group["ppe_summary"] and detection.get("ppe_summary"):
                group["ppe_summary"] = detection["ppe_summary"]
            state_priority = {"ppe_ok": 0, "unclear": 1, "violation": 2}
            detection_status = str(detection.get("person_ppe_status") or "violation")
            if state_priority.get(detection_status, 2) > state_priority.get(group["person_ppe_status"], 2):
                group["person_ppe_status"] = detection_status
            group["needs_review"] = group["needs_review"] or bool(detection.get("needs_review", True))
            for key in (
                "person_confidence",
                "helmet_confidence",
                "vest_confidence",
                "no_helmet_confidence",
                "no_vest_confidence",
            ):
                value = detection.get(key)
                current = group.get(key)
                if value is not None and (current is None or float(value) > float(current)):
                    group[key] = float(value)
        return sorted(groups.values(), key=lambda group: float(group["confidence"]), reverse=True)

    @staticmethod
    def ppe_summary(safe: dict[str, float], violations: list[dict[str, Any]]) -> str:
        violation_conf: dict[str, float] = {}
        for item in violations:
            label = str(item["label"])
            violation_conf[label] = max(violation_conf.get(label, 0.0), float(item["confidence"]))

        def state_text(title: str, ok_label: str, missing_label: str) -> str:
            missing_conf = violation_conf.get(missing_label, 0.0)
            ok_conf = safe.get(ok_label, 0.0)
            if missing_conf > 0:
                return f"{title}: немає ({missing_conf:.2f})"
            if ok_conf > 0:
                return f"{title}: є ({ok_conf:.2f})"
            return f"{title}: не визначено"

        return "; ".join(
            [
                state_text("Каска", "HELMET", "NO_HELMET"),
                state_text("Жилет", "VEST", "NO_VEST"),
            ]
        )

    @staticmethod
    def risk_score(violations: list[dict[str, Any]], ppe_status: str) -> float:
        if ppe_status == "ppe_ok":
            return 0.0
        if not violations:
            return 0.25
        score = max(float(item["confidence"]) for item in violations)
        missing_items = {str(item["label"]) for item in violations}
        if {"NO_HELMET", "NO_VEST"}.issubset(missing_items):
            score += 0.15
        return min(1.0, score)

    @staticmethod
    def build_person_checks(items: list[dict[str, Any]], persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[Any, dict[str, Any]] = {}
        for index, person in enumerate(persons):
            key = ("person_box", person["box"])
            groups.setdefault(
                key,
                {
                    "box": person["box"],
                    "track_id": person.get("track_id"),
                    "person_confidence": float(person["confidence"]),
                    "safe": {},
                    "violations": [],
                },
            )

        for index, item in enumerate(items):
            if item["is_person"]:
                continue
            key = Detector.person_group_key(item, index)
            box = item.get("person_box") or item["box"]
            group = groups.setdefault(
                key,
                {
                    "box": box,
                    "track_id": item.get("track_id"),
                    "person_confidence": 0.0,
                    "safe": {},
                    "violations": [],
                },
            )
            group["box"] = Detector.union_boxes(group["box"], box)
            if group["track_id"] is None and item.get("track_id") is not None:
                group["track_id"] = item["track_id"]
            if item["is_safe"]:
                current = group["safe"].get(item["label"], 0.0)
                group["safe"][item["label"]] = max(current, float(item["confidence"]))
            elif item["is_violation"]:
                group["violations"].append(item)

        checks: list[dict[str, Any]] = []
        for group in groups.values():
            safe = group["safe"]
            kept: list[dict[str, Any]] = []
            for item in group["violations"]:
                confidence = float(item["confidence"])
                if item["label"] == "NO_HELMET" and safe.get("HELMET", 0.0) >= confidence - SAFE_CONFLICT_MARGIN:
                    continue
                if item["label"] == "NO_VEST" and safe.get("VEST", 0.0) >= confidence - SAFE_CONFLICT_MARGIN:
                    continue
                kept.append(item)

            summary = Detector.ppe_summary(safe, kept)
            if kept:
                labels = sorted({str(item["label"]) for item in kept})
                label = "+".join(labels)
                confidence = max(float(item["confidence"]) for item in kept)
                ppe_status = "violation"
                needs_review = True
            elif safe.get("HELMET", 0.0) > 0 and safe.get("VEST", 0.0) > 0:
                label = "PPE_OK"
                confidence = max(float(group["person_confidence"]), safe.get("HELMET", 0.0), safe.get("VEST", 0.0))
                ppe_status = "ppe_ok"
                needs_review = False
            else:
                label = "PPE_UNCLEAR"
                confidence = max(float(group["person_confidence"]), *(float(value) for value in safe.values()), 0.0)
                ppe_status = "unclear"
                needs_review = True
            violation_confidence = {
                item["label"]: max(
                    float(item["confidence"]),
                    max(
                        (float(previous["confidence"]) for previous in kept if previous["label"] == item["label"]),
                        default=0.0,
                    ),
                )
                for item in kept
            }

            checks.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": group["box"],
                    "person_box": group["box"],
                    "track_id": group["track_id"],
                    "object_track_id": None,
                    "ppe_summary": summary,
                    "person_ppe_status": ppe_status,
                    "risk_score": Detector.risk_score(kept, ppe_status),
                    "person_confidence": float(group["person_confidence"]) if group["person_confidence"] else None,
                    "helmet_confidence": safe.get("HELMET"),
                    "vest_confidence": safe.get("VEST"),
                    "no_helmet_confidence": violation_confidence.get("NO_HELMET"),
                    "no_vest_confidence": violation_confidence.get("NO_VEST"),
                    "needs_review": needs_review,
                }
            )
        return checks

    @staticmethod
    def draw_person_checks(
        draw,
        detections: list[dict[str, Any]],
        only_violations: bool = False,
        show_annotations: bool = True,
    ) -> None:
        for group in Detector.group_person_checks(detections):
            if only_violations and group.get("person_ppe_status") != "violation":
                continue
            x1, y1, x2, y2 = group["box"]
            prefix = f"#{group['track_id']} " if group["track_id"] is not None else ""
            if group.get("person_ppe_status") == "ppe_ok":
                color = (0, 180, 0)
            elif group.get("person_ppe_status") == "unclear":
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)
            if show_annotations:
                cv2.putText(
                    draw,
                    f"{prefix}{group['confidence']:.2f}",
                    (x1, max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

    @staticmethod
    def union_boxes(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    @staticmethod
    def match_person(violation_box: tuple[int, int, int, int], persons: list[dict[str, Any]]) -> dict[str, Any] | None:
        vx1, vy1, vx2, vy2 = violation_box
        cx = (vx1 + vx2) / 2
        cy = (vy1 + vy2) / 2
        violation_area = max(1, (vx2 - vx1) * (vy2 - vy1))
        best_person = None
        best_score = 0.0
        for person in persons:
            px1, py1, px2, py2 = person["box"]
            ix1, iy1 = max(vx1, px1), max(vy1, py1)
            ix2, iy2 = min(vx2, px2), min(vy2, py2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            contains_center = px1 <= cx <= px2 and py1 <= cy <= py2
            score = intersection / violation_area
            if contains_center:
                score += 1.0
            if score > best_score:
                best_score = score
                best_person = person
        return best_person if best_score > 0 else None


class VideoWorker(QObject):
    frame_ready = Signal(bytes)
    status = Signal(str)
    incident_created = Signal()
    finished = Signal()

    def __init__(
        self,
        camera_id: str,
        source: str,
        detector: Detector,
        store: Store,
        shift_id: int | None,
        user_id: int | None,
        box_mode_getter,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.store = store
        self.shift_id = shift_id
        self.user_id = user_id
        self.box_mode_getter = box_mode_getter
        self.stop_requested = False
        self.slow_warning_emitted = False
        self.run_id = uuid.uuid4().hex[:8]
        self.person_review_times: dict[tuple[Any, ...], float] = {}
        self.open_incidents_by_key: dict[tuple[Any, ...], list[int]] = {}

    def stop(self) -> None:
        self.stop_requested = True

    def log_event(self, event_type: str, message: str) -> None:
        self.store.log_event(self.shift_id, self.user_id, event_type, message)

    def source_name(self) -> str:
        return Path(self.source).name

    def source_description(self) -> str:
        return f"{camera_label(self.camera_id)} / {self.source_name()}"

    @staticmethod
    def resize_for_processing(frame, max_width: int):
        if frame.shape[1] <= max_width:
            return frame
        scale = max_width / frame.shape[1]
        return cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            message = f"{camera_label(self.camera_id)}: не вдалося відкрити відеофайл {self.source_name()}"
            self.status.emit(message)
            self.log_event("source_error", message)
            self.finished.emit()
            return

        self.log_event("source_start", f"Запущено демо-аналіз: {self.source_description()}, номер запуску={self.run_id}")
        try:
            self.run_file(cap)
        except Exception as ex:
            message = f"Помилка аналізу {self.source_description()}: {ex}"
            self.status.emit(message)
            self.log_event("system_error", message)
        finally:
            cap.release()
            message = f"{self.source_description()}: аналіз завершено"
            self.status.emit(message)
            self.log_event("source_stop", message)
            self.finished.emit()

    def analyze_video_timed(self, frame) -> tuple[Any, list[dict[str, Any]]]:
        started = time.perf_counter()
        draw, detections = self.detector.analyze_video(frame, self.box_mode_getter())
        elapsed = time.perf_counter() - started
        if elapsed >= MODEL_SLOW_WARNING_SEC and not self.slow_warning_emitted:
            self.slow_warning_emitted = True
            message = f"Модель обробляла кадр {elapsed:.2f} с; демо-аналіз продовжується в ощадному режимі."
            self.status.emit(message)
            self.log_event("model_slow", message)
        return draw, detections

    def run_file(self, cap) -> None:
        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        self.detector.reset_tracking()
        self.status.emit(
            f"{self.source_description()}: аналізуються кожні {FILE_PROCESS_EVERY_N} кадрів"
        )
        while cap.isOpened() and not self.stop_requested:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            frame = self.resize_for_processing(frame, VIDEO_MAX_WIDTH)

            detections: list[dict[str, Any]] = []
            draw = frame
            did_analyze = False
            if frame_index % FILE_PROCESS_EVERY_N == 0:
                draw, detections = self.analyze_video_timed(frame)
                did_analyze = True

            video_time_sec = frame_index / fps
            if did_analyze:
                self.resolve_equipped_people(frame, detections, frame_index)
            new_groups = self.new_incident_groups(detections, frame_index, video_time_sec)
            if did_analyze or frame_index == 1:
                self.frame_ready.emit(frame_to_jpg(draw))
            for group in new_groups:
                self.create_incident(frame, draw, frame_index, group)
                self.incident_created.emit()

            if frame_index % 60 == 0:
                progress = f"/{total_frames}" if total_frames else ""
                self.status.emit(f"{self.source_description()}: кадр {frame_index}{progress}")

    def violation_key(self, detection: dict[str, Any]) -> tuple[str, str, int] | None:
        track_id = detection.get("track_id")
        if track_id is None:
            return None
        return (self.camera_id, self.source_name(), int(track_id))

    def review_group_key(self, group: dict[str, Any]) -> tuple[Any, ...] | None:
        track_id = group.get("track_id")
        if track_id is not None:
            return ("track", self.camera_id, self.source_name(), int(track_id))
        box = group.get("box")
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return ("box", self.camera_id, self.source_name(), int((x1 + x2) // 200), int((y1 + y2) // 200))

    def new_incident_groups(
        self,
        detections: list[dict[str, Any]],
        frame_index: int,
        video_time_sec: float,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for group in Detector.group_person_checks(detections):
            needs_review = bool(group.get("needs_review", True))
            key = self.review_group_key(group)
            if key is None:
                key = ("untracked", self.camera_id, self.source_name())
            if not needs_review:
                continue
            last_review_time = self.person_review_times.get(key)
            if last_review_time is not None and video_time_sec - last_review_time < PERSON_REVIEW_COOLDOWN_SEC:
                continue
            self.person_review_times[key] = video_time_sec
            group["_review_key"] = key
            candidates.append(group)

        return candidates

    def resolve_equipped_people(self, raw_frame, detections: list[dict[str, Any]], frame_index: int) -> None:
        for group in Detector.group_person_checks(detections):
            if group.get("person_ppe_status") != "ppe_ok":
                continue
            key = self.review_group_key(group)
            if key is None:
                continue
            incident_ids = self.open_incidents_by_key.pop(key, [])
            if not incident_ids:
                continue
            track_id = int(group["track_id"]) if group.get("track_id") is not None else None
            ppe_summary = str(group.get("ppe_summary") or "")
            track_part = f"_P{track_id}" if track_id is not None else ""
            name = f"{self.camera_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{track_part}_PPE_OK.jpg"
            screenshot = FOLDERS["resolved"] / name
            resolved_frame = raw_frame.copy()
            Detector.draw_person_checks(resolved_frame, group["detections"], show_annotations=False)
            write_frame_atomic(screenshot, resolved_frame)
            was_saved = False
            try:
                for incident_id in incident_ids:
                    if self.store.mark_incident_equipped(incident_id, frame_index, str(screenshot), ppe_summary):
                        was_saved = True
                        self.log_event(
                            "ppe_resolved",
                            f"{camera_label(self.camera_id)}: інцидент #{incident_id}, номер людини {track_id if track_id is not None else 'не визначено'} екіпірувався на кадрі {frame_index}, ЗІЗ={ppe_summary}",
                        )
            finally:
                if not was_saved:
                    remove_file_if_exists(screenshot)
            self.incident_created.emit()

    def create_incident(self, raw_frame, draw_frame, frame_index: int, group: dict[str, Any]) -> int | None:
        detections = group.get("detections", [])
        if not detections:
            return None
        best = max(detections, key=lambda item: item["confidence"])
        key = self.violation_key(best)
        review_key = group.get("_review_key") or self.review_group_key(group)
        track_id = int(best["track_id"]) if best.get("track_id") is not None else None
        same_person = detections
        labels = sorted({str(item["label"]) for item in same_person})
        label = "+".join(labels)
        risk_score = float(group.get("risk_score", 0.0))
        ppe_summary = next((str(item["ppe_summary"]) for item in same_person if item.get("ppe_summary")), "")
        track_part = f"_P{track_id}" if track_id is not None else ""
        name = f"{self.camera_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{track_part}_{label}.jpg"
        screenshot = FOLDERS["fixed_new"] / name
        raw = FOLDERS["raw"] / name
        incident_frame = raw_frame.copy()
        Detector.draw_person_checks(incident_frame, same_person, show_annotations=False)
        try:
            write_frame_atomic(screenshot, incident_frame)
            write_frame_atomic(raw, raw_frame)
            incident_id = self.store.add_incident(
                camera_id=self.camera_id,
                source_file=self.source_name(),
                frame_index=frame_index,
                label=label,
                person_ppe_status=str(group.get("person_ppe_status") or "violation"),
                risk_score=risk_score,
                original_screenshot_path=str(screenshot),
                raw_path=str(raw),
                shift_id=self.shift_id,
                track_id=track_id,
                violation_key=(
                    f"{self.camera_id}:{self.source_name()}:{self.run_id}:track:{track_id}:frame:{frame_index}"
                    if key
                    else f"{self.camera_id}:{self.source_name()}:{self.run_id}:untracked:frame:{frame_index}"
                ),
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
        if review_key is not None and group.get("person_ppe_status") != "ppe_ok":
            self.open_incidents_by_key.setdefault(review_key, []).append(incident_id)
        self.log_event(
            "alert",
            f"Створено картку перевірки #{incident_id}: камера={camera_label(self.camera_id)}, джерело={self.source_name()}, кадр={frame_index}, висновок моделі={detection_label(label)}, пріоритет перевірки={risk_score:.4f}, номер людини={track_id if track_id is not None else 'не визначено'}, ЗІЗ={ppe_summary}",
        )
        id_text = f"номер людини {track_id}" if track_id is not None else "номер людини не визначено"
        self.status.emit(f"{camera_label(self.camera_id)}: створено картку перевірки #{incident_id}: {id_text}, {detection_label(label)}")
        return incident_id


