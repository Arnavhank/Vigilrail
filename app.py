import mimetypes
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from ultralytics import YOLO
import yt_dlp
from werkzeug.utils import secure_filename

from reid_integration import DictMemoryReIDTracker, get_reid_extractor_cached


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(BASE_DIR, "best.pt")
MODEL2_PATH = os.path.join(BASE_DIR, "best2.pt")
DEFAULT_STREAM_URL = "https://www.youtube.com/watch?v=t65TCpcJOPQ"
CONF_THRESHOLD = 0.35

COLOR_MODEL_1 = (0, 0, 255)      # Red (BGR)
COLOR_MODEL_2 = (0, 165, 255)    # Orange (BGR)
COLOR_REID = (0, 230, 100)       # Green-cyan for ReID track labels (BGR)

# Clips that use hybrid IoU + embedding (OSNet ONNX optional) + EMA centroid
REID_STYLE_FILES = frozenset({"reid_demo.mp4"})


def _is_person_like(label: str) -> bool:
    s = label.lower()
    return "person" in s or "pedestrian" in s or "human" in s


def _draw_reid_panel(frame: np.ndarray, lines: list[str]) -> None:
    x1, y1 = 10, 10
    line_h = 22
    pad = 8
    w = min(420, max(0, frame.shape[1] - 20))
    h = pad * 2 + len(lines) * line_h
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (40, 44, 62), -1)
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (120, 190, 255), 1)
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line[:70],
            (x1 + pad, y1 + pad + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 240, 250),
            1,
            cv2.LINE_AA,
        )


app = Flask(__name__, static_folder=BASE_DIR)
app.config["SECRET_KEY"] = "vigilrail-live-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.after_request
def _cors_all(resp):
    """Allow alerts page on another port (or file) to call API and fetch /status."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


class DualModelStreamProcessor:
    def __init__(self) -> None:
        self.model1 = None
        self.model2 = None
        self.model1_loaded = False
        self.model2_loaded = False

        self.stream_url = DEFAULT_STREAM_URL
        self.stream_running = False
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.alert_cooldown = defaultdict(float)

        self._load_models()

    def _load_models(self) -> None:
        try:
            self.model1 = YOLO(MODEL1_PATH)
            self.model1_loaded = True
        except Exception as exc:
            print(f"[WARN] Failed to load model1 ({MODEL1_PATH}): {exc}")

        try:
            self.model2 = YOLO(MODEL2_PATH)
            self.model2_loaded = True
        except Exception as exc:
            print(f"[WARN] Failed to load model2 ({MODEL2_PATH}): {exc}")

    @staticmethod
    def _resolve_stream_url(youtube_url: str) -> str:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            resolved = info.get("url")
            if not resolved:
                raise RuntimeError("Unable to resolve stream URL via yt-dlp.")
            return resolved

    @staticmethod
    def _extract_detections(results, model_tag: str):
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": label,
                        "conf": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "model": model_tag,
                    }
                )
        return detections

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det.get("track_id")
            if tid is not None:
                color = COLOR_REID
                label = f"Person ID {tid} · {det['label']} {det['conf']:.2f}"
            else:
                color = COLOR_MODEL_1 if det["model"] == "model_1" else COLOR_MODEL_2
                label = f"{det['label']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def _emit_alerts(self, detections, extra=None):
        now = time.time()
        for det in detections:
            key = f"{det['model']}:{det['label']}"
            if det.get("track_id") is not None:
                key += f":id{det['track_id']}"
            if now - self.alert_cooldown[key] < 1.5:
                continue
            self.alert_cooldown[key] = now
            payload = {
                "model": det["model"],
                "label": det["label"],
                "confidence": round(det["conf"], 3),
                "severity": "high" if det["model"] == "model_1" else "medium",
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            if det.get("track_id") is not None:
                payload["person_id"] = int(det["track_id"])
                payload["event_label"] = f"Person ID {det['track_id']}"
            if extra:
                payload.update(extra)
            socketio.emit("alert_event", payload)

    def start(self, youtube_url: str) -> None:
        self.stream_url = youtube_url
        if self.stream_running:
            self.stop()
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.stream_running = True
        self.worker_thread.start()
        self.emit_status()

    def stop(self) -> None:
        self.stop_event.set()
        self.stream_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        self.worker_thread = None
        self.emit_status()

    def _run_pipeline(self) -> None:
        cap = None
        try:
            resolved_stream = self._resolve_stream_url(self.stream_url)
            cap = cv2.VideoCapture(resolved_stream)
            if not cap.isOpened():
                raise RuntimeError("OpenCV failed to open resolved stream URL.")

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                merged = []
                if self.model1_loaded:
                    r1 = self.model1.predict(source=frame, verbose=False)
                    merged.extend(self._extract_detections(r1, "model_1"))

                if self.model2_loaded:
                    r2 = self.model2.predict(source=frame, verbose=False)
                    merged.extend(self._extract_detections(r2, "model_2"))

                if merged:
                    self._emit_alerts(merged)

                annotated = self._draw_detections(frame, merged)
                ok_enc, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok_enc:
                    continue

                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
        except Exception as exc:
            print(f"[ERROR] Stream pipeline stopped: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.stream_running = False
            self.emit_status()

    def mjpeg_generator(self):
        while True:
            if not self.stream_running:
                blank = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Stream offline - start stream from UI",
                    (180, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
                ok_enc, jpeg = cv2.imencode(".jpg", blank)
                if ok_enc:
                    frame_bytes = jpeg.tobytes()
                else:
                    frame_bytes = b""
            else:
                with self.lock:
                    frame_bytes = self.latest_jpeg

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.03)

    def emit_status(self):
        status_payload = {
            "running": self.stream_running,
            "model1_loaded": self.model1_loaded,
            "model2_loaded": self.model2_loaded,
        }
        socketio.emit("status", status_payload)

    def status_json(self):
        return {
            "running": self.stream_running,
            "model1_loaded": self.model1_loaded,
            "model2_loaded": self.model2_loaded,
            "stream_url": self.stream_url,
            "reid_onnx_loaded": get_reid_extractor_cached(BASE_DIR) is not None,
        }


processor = DualModelStreamProcessor()


@app.get("/")
def root():
    return send_from_directory(BASE_DIR, "live.html")


@app.get("/alerts.html")
def alerts_page():
    return send_from_directory(BASE_DIR, "alerts.html")


@app.get("/live.css")
def live_css():
    return send_from_directory(BASE_DIR, "live.css")


@app.get("/live.js")
def live_js():
    return send_from_directory(BASE_DIR, "live.js")


@app.get("/video_feed")
def video_feed():
    return Response(
        processor.mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start_stream")
def start_stream():
    data = request.get_json(silent=True) or {}
    stream_url = data.get("stream_url", "").strip() or DEFAULT_STREAM_URL
    try:
        processor.start(stream_url)
        return jsonify({"ok": True, "message": "Stream started."})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/stop_stream")
def stop_stream():
    processor.stop()
    return jsonify({"ok": True})


@app.get("/status")
def status():
    return jsonify(processor.status_json())


@app.get("/clips/<path:filename>")
def serve_clip(filename):
    """Serve raw MP4/WebM for <video> previews (same files processed by /process_video)."""
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name:
        abort(404)
    path = os.path.join(BASE_DIR, safe_name)
    if not os.path.isfile(path):
        abort(404)
    mtype, _ = mimetypes.guess_type(safe_name)
    if not mtype or not mtype.startswith("video/"):
        mtype = "video/mp4"
    return send_from_directory(BASE_DIR, safe_name, mimetype=mtype, conditional=True)


@socketio.on("connect")
def on_connect():
    processor.emit_status()


@app.route("/process_video/<path:video_name>")
def process_video(video_name):
    safe_name = secure_filename(os.path.basename(video_name))
    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid video name."}), 400
    return Response(
        generate_video(safe_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _error_jpeg(message: str):
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        message[:80],
        (24, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    ok, jpeg = cv2.imencode(".jpg", blank)
    return jpeg.tobytes() if ok else b""


def generate_video(safe_filename: str):
    video_path = os.path.join(BASE_DIR, safe_filename)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        err = _error_jpeg(f"Video not found: {safe_filename}")
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + err + b"\r\n"
            )
            time.sleep(0.5)

    cap = cv2.VideoCapture(video_path)
    use_reid_style = safe_filename.lower() in REID_STYLE_FILES
    reid_extractor = get_reid_extractor_cached(BASE_DIR) if use_reid_style else None
    dict_tracker: Optional[DictMemoryReIDTracker] = None

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            merged = []

            if use_reid_style:
                if processor.model1_loaded:
                    r1 = processor.model1(frame, verbose=False)[0]
                    merged = processor._extract_detections([r1], "model_1")
                persons = [d for d in merged if _is_person_like(d["label"])]
                if not persons and merged:
                    persons = list(merged)
                if dict_tracker is None:
                    dict_tracker = DictMemoryReIDTracker(reid_extractor)
                persons = dict_tracker.update(frame, persons)
                merged = persons

                if merged:
                    processor._emit_alerts(
                        merged,
                        extra={
                            "source": "alert_clip",
                            "file": safe_filename,
                            "reid_active": True,
                            "pipeline": "yolo_dict_reid_memory",
                        },
                    )

                frame = processor._draw_detections(frame, merged)
                onnx_ok = reid_extractor is not None
                if merged:
                    lines = [
                        "ReID · prev_tracks + cosine + px dist",
                        "OSNet: " + ("on" if onnx_ok else "off (spatial only)"),
                    ]
                    for d in merged[:4]:
                        lines.append(
                            f"Person ID {d['track_id']}: {d['label']} ({d['conf']:.2f})"
                        )
                    _draw_reid_panel(frame, lines[:6])
                else:
                    _draw_reid_panel(
                        frame,
                        [
                            "ReID · dict memory tracker",
                            "No detections this frame",
                        ],
                    )
            else:
                if processor.model1_loaded:
                    r1 = processor.model1(frame, verbose=False)[0]
                    merged.extend(processor._extract_detections([r1], "model_1"))

                if processor.model2_loaded:
                    r2 = processor.model2(frame, verbose=False)[0]
                    merged.extend(processor._extract_detections([r2], "model_2"))

                if merged:
                    processor._emit_alerts(
                        merged,
                        extra={"source": "alert_clip", "file": safe_filename},
                    )

                frame = processor._draw_detections(frame, merged)

            ok_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok_enc:
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)

