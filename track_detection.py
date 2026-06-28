
import subprocess, sys

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("ultralytics>=8.0.0", "deep-sort-realtime>=1.3.2",
            "opencv-python-headless>=4.8.0", "numpy>=1.24.0")

print("Dependencies installed.")

import cv2, math, time, os, sys, logging, warnings
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


warnings.filterwarnings("ignore")

class Config:
    YOLO_MODEL_PATH    = "best.pt"
    CONF_THRESHOLD     = 0.45
    CLASS_NAMES        = ["person", "bag", "weapon"]

    MAX_AGE            = 30
    N_INIT             = 3
    MAX_IOU_DISTANCE   = 0.7
    MAX_COSINE_DISTANCE= 0.3
    NN_BUDGET          = 100

    PLATFORM_EDGE_ZONE = [(0.0, 0.75), (1.0, 0.75), (1.0, 1.0), (0.0, 1.0)]
    RESTRICTED_ZONES   = [
        [(0.8, 0.0), (1.0, 0.0), (1.0, 0.4), (0.8, 0.4)],
    ]

    LOITERING_SECONDS      = 30
    UNATTENDED_BAG_SECONDS = 20
    UNATTENDED_BAG_DIST    = 150
    FALLEN_RATIO           = 1.6
    CROWD_THRESHOLD        = 8
    RUNNING_SPEED          = 18
    PROXIMITY_DIST         = 60
    TRAJECTORY_HISTORY     = 60

    ALERT_COOLDOWN    = 5
    SAVE_ALERT_FRAMES = True
    ALERT_OUTPUT_DIR  = "alerts/"

    DATASET_YAML = "dataset.yaml"


# STEP 3: Helpers
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1+x2)/2), int((y1+y2)/2))

def point_in_polygon(point, polygon):
    result = cv2.pointPolygonTest(
        np.array(polygon, dtype=np.float32), (float(point[0]), float(point[1])), False)
    return result >= 0

def scale_polygon(polygon, frame_w, frame_h):
    return [(int(x*frame_w), int(y*frame_h)) for x, y in polygon]


# STEP 4: Behavior Engine
class BehaviorEngine:
    def __init__(self):
        self.first_seen       = {}
        self.last_seen        = {}
        self.trajectories     = defaultdict(list)
        self.prev_centers     = {}
        self.flagged_ids      = set()
        self.stationary_since = {}
        self.bag_status       = {}

    def analyze(self, active_tracks, frame, timestamp):
        h, w = frame.shape[:2]
        alerts = []

        persons = {tid: t for tid, t in active_tracks.items() if t["class"] == "person"}
        bags    = {tid: t for tid, t in active_tracks.items() if t["class"] == "bag"}
        weapons = {tid: t for tid, t in active_tracks.items() if t["class"] == "weapon"}

        for tid, t in active_tracks.items():
            center = bbox_center(t["bbox"])
            self.trajectories[tid].append(center)
            if len(self.trajectories[tid]) > Config.TRAJECTORY_HISTORY:
                self.trajectories[tid].pop(0)
            if tid not in self.first_seen:
                self.first_seen[tid] = timestamp
            self.last_seen[tid] = timestamp

        for bid, bag in bags.items():
            bag_center = bbox_center(bag["bbox"])
            nearest = min(
                (euclidean(bag_center, bbox_center(p["bbox"])) for p in persons.values()),
                default=float("inf")
            )
            self.bag_status[bid] = "attended" if nearest <= Config.UNATTENDED_BAG_DIST else "unattended"

        alerts += self._check_weapon(weapons)
        alerts += self._check_loitering(persons, timestamp, w, h)
        alerts += self._check_fallen(persons)
        alerts += self._check_running(persons)
        alerts += self._check_unattended_bag(bags, persons, timestamp)
        alerts += self._check_platform_edge(persons, w, h)
        alerts += self._check_restricted_zone(persons, w, h)
        alerts += self._check_overcrowding(persons, w, h)
        alerts += self._check_tailgating(persons)
        alerts += self._check_wrong_direction(persons)

        self.flagged_ids = {a["id"] for a in alerts if "id" in a}
        return alerts

    def _check_weapon(self, weapons):
        return [{"type": "WEAPON DETECTED", "id": tid,
                 "priority": "CRITICAL", "bbox": t["bbox"]}
                for tid, t in weapons.items()]

    def _check_loitering(self, persons, timestamp, w, h):
        alerts = []
        for tid, t in persons.items():
            dwell = timestamp - self.first_seen.get(tid, timestamp)
            if dwell > Config.LOITERING_SECONDS:
                alerts.append({"type": "LOITERING", "id": tid,
                                "duration": round(dwell, 1),
                                "priority": "HIGH", "bbox": t["bbox"]})
        return alerts

    def _check_fallen(self, persons):
        alerts = []
        for tid, t in persons.items():
            x1, y1, x2, y2 = t["bbox"]
            bw, bh = x2-x1, y2-y1
            if bh > 0 and (bw/bh) > Config.FALLEN_RATIO:
                alerts.append({"type": "FALLEN PERSON", "id": tid,
                                "priority": "CRITICAL", "bbox": t["bbox"]})
        return alerts

    def _check_running(self, persons):
        alerts = []
        for tid, t in persons.items():
            center = bbox_center(t["bbox"])
            if tid in self.prev_centers:
                speed = euclidean(center, self.prev_centers[tid])
                if speed > Config.RUNNING_SPEED:
                    alerts.append({"type": "RUNNING/PANIC", "id": tid,
                                   "speed": round(speed, 1),
                                   "priority": "MEDIUM", "bbox": t["bbox"]})
            self.prev_centers[tid] = center
        return alerts

    def _check_unattended_bag(self, bags, persons, timestamp):
        alerts = []
        for bid, bag in bags.items():
            bag_center = bbox_center(bag["bbox"])
            nearest_dist = min(
                (euclidean(bag_center, bbox_center(p["bbox"])) for p in persons.values()),
                default=float("inf")
            )
            if nearest_dist > Config.UNATTENDED_BAG_DIST:
                if bid not in self.stationary_since:
                    self.stationary_since[bid] = timestamp
                alone_for = timestamp - self.stationary_since[bid]
                if alone_for > Config.UNATTENDED_BAG_SECONDS:
                    alerts.append({"type": "UNATTENDED BAG", "id": bid,
                                   "alone_for": round(alone_for, 1),
                                   "priority": "HIGH", "bbox": bag["bbox"]})
            else:
                self.stationary_since.pop(bid, None)
        return alerts

    def _check_platform_edge(self, persons, w, h):
        alerts = []
        edge_poly = scale_polygon(Config.PLATFORM_EDGE_ZONE, w, h)
        for tid, t in persons.items():
            if point_in_polygon(bbox_center(t["bbox"]), edge_poly):
                alerts.append({"type": "PLATFORM EDGE INTRUSION", "id": tid,
                                "priority": "CRITICAL", "bbox": t["bbox"]})
        return alerts

    def _check_restricted_zone(self, persons, w, h):
        alerts = []
        for zone_norm in Config.RESTRICTED_ZONES:
            zone = scale_polygon(zone_norm, w, h)
            for tid, t in persons.items():
                if point_in_polygon(bbox_center(t["bbox"]), zone):
                    alerts.append({"type": "RESTRICTED ZONE", "id": tid,
                                   "priority": "HIGH", "bbox": t["bbox"]})
        return alerts

    def _check_overcrowding(self, persons, w, h):
        if len(persons) > Config.CROWD_THRESHOLD:
            return [{"type": "OVERCROWDING",
                     "count": len(persons),
                     "priority": "MEDIUM"}]
        return []

    def _check_tailgating(self, persons):
        alerts = []
        ids = list(persons.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                d = euclidean(bbox_center(persons[a]["bbox"]),
                              bbox_center(persons[b]["bbox"]))
                if d < Config.PROXIMITY_DIST:
                    alerts.append({"type": "TAILGATING/PROXIMITY",
                                   "id": a, "partner": b,
                                   "dist": round(d, 1),
                                   "priority": "LOW",
                                   "bbox": persons[a]["bbox"]})
        return alerts

    def _check_wrong_direction(self, persons):
        alerts = []
        vectors = []
        for tid, t in persons.items():
            traj = self.trajectories.get(tid, [])
            if len(traj) >= 5:
                dx = traj[-1][0] - traj[-5][0]
                dy = traj[-1][1] - traj[-5][1]
                vectors.append((tid, np.array([dx, dy]), t["bbox"]))

        if len(vectors) < 3:
            return alerts

        avg_vec = np.mean([v for _, v, _ in vectors], axis=0)
        avg_norm = np.linalg.norm(avg_vec)
        if avg_norm < 1e-6:
            return alerts
        avg_unit = avg_vec / avg_norm

        for tid, vec, bbox in vectors:
            dot = float(np.dot(vec, avg_unit))
            if dot < -200:
                alerts.append({"type": "WRONG DIRECTION", "id": tid,
                                "dot": round(dot, 1),
                                "priority": "LOW", "bbox": bbox})
        return alerts


class AlertSystem:
    def __init__(self):
        os.makedirs(Config.ALERT_OUTPUT_DIR, exist_ok=True)
        logging.basicConfig(
            filename="alerts.log",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.last_alert_time = defaultdict(float)

    def trigger(self, alert, frame, frame_id):
        alert_type = alert["type"]
        alert_id   = alert.get("id", "global")
        key        = f"{alert_type}_{alert_id}"
        now        = time.time()

        if now - self.last_alert_time[key] < Config.ALERT_COOLDOWN:
            return
        self.last_alert_time[key] = now

        priority = alert.get("priority", "MEDIUM")
        extra    = {k: v for k, v in alert.items()
                    if k not in ("type", "id", "priority", "bbox")}
        msg      = f"{alert_type} | ID:{alert_id} | {extra}"

        if priority == "CRITICAL":
            logging.critical(msg)
        elif priority == "HIGH":
            logging.warning(msg)
        else:
            logging.info(msg)

        print(f"[{priority}] {msg}")

        if Config.SAVE_ALERT_FRAMES:
            snap = frame.copy()
            if "bbox" in alert:
                x1, y1, x2, y2 = map(int, alert["bbox"])
                cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(snap, alert_type, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ts   = time.strftime("%Y%m%d_%H%M%S")
            tag  = alert_type.replace(" ", "_").replace("/", "_")
            path = os.path.join(Config.ALERT_OUTPUT_DIR,
                                f"{tag}_id{alert_id}_f{frame_id}_{ts}.jpg")
            cv2.imwrite(path, snap)


def draw(frame, tracks, alerts, engine, bag_status):
    vis = frame.copy()
    h, w = vis.shape[:2]

    COLOR = {
        "person":         (50,  220,  50),
        "bag":            (255, 165,   0),
        "attended_bag":   (255, 165,   0),
        "unattended_bag": (0,   0,   255),
        "weapon":         (0,   0,   255),
    }

    edge = [(int(x*w), int(y*h)) for x, y in Config.PLATFORM_EDGE_ZONE]
    cv2.polylines(vis, [np.array(edge)], True, (0, 100, 255), 2)
    cv2.putText(vis, "TRACK ZONE", edge[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    for zone_norm in Config.RESTRICTED_ZONES:
        zone = [(int(x*w), int(y*h)) for x, y in zone_norm]
        cv2.polylines(vis, [np.array(zone)], True, (0, 0, 200), 2)
        cv2.putText(vis, "RESTRICTED", zone[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    for tid, t in tracks.items():
        x1, y1, x2, y2 = map(int, t["bbox"])
        cls     = t["class"]
        flagged = tid in engine.flagged_ids

        if cls == "bag":
            status = bag_status.get(tid, "attended")
            label  = f"{'UN' if status == 'unattended' else ''}ATTENDED BAG #{tid}"
            color  = COLOR["unattended_bag"] if status == "unattended" else COLOR["attended_bag"]
        else:
            label = f"{cls.upper()} #{tid}"
            color = (0, 0, 255) if flagged else COLOR.get(cls, (200, 200, 200))

        thick = 3 if flagged else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        traj = engine.trajectories.get(tid, [])
        for i in range(1, len(traj)):
            cv2.line(vis, traj[i-1], traj[i], color, 1)

    if alerts:
        panel_h = min(len(alerts)*28 + 10, 220)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (500, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
        y = 24
        for alert in alerts[-7:]:
            pri = alert.get("priority", "MEDIUM")
            clr = (0, 0, 255)   if pri == "CRITICAL" else \
                  (0, 165, 255) if pri == "HIGH"     else (0, 220, 220)
            txt = f"[{pri}] {alert['type']}  ID:{alert.get('id','')}"
            cv2.putText(vis, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
            y += 28

    return vis


def parse_detections(results):
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = results.names[cls]
        w_box = x2 - x1
        h_box = y2 - y1
        detections.append(([x1, y1, w_box, h_box], conf, label))
    return detections


def ensure_dataset_yaml():
    if not os.path.exists(Config.DATASET_YAML):
        yaml_content = (
            "# dataset.yaml - auto-generated\n"
            "path: ./dataset\n"
            "train: images/train\n"
            "val:   images/val\n"
            "test:  images/test\n\n"
            "nc: 3\n"
            "names:\n"
            "  0: person\n"
            "  1: bag\n"
            "  2: weapon\n"
        )
        with open(Config.DATASET_YAML, "w") as f:
            f.write(yaml_content)
        print(f"{Config.DATASET_YAML} created - update paths before training.")


def train_model():
    ensure_dataset_yaml()
    model = YOLO("yolov8n.pt")
    model.train(
        data      = Config.DATASET_YAML,
        epochs    = 100,
        imgsz     = 640,
        batch     = 16,
        name      = "railway_surveillance",
        patience  = 20,
        augment   = True,
        mosaic    = 1.0,
        degrees   = 5.0,
        flipud    = 0.1,
    )
    model.val()
    model.export(format="onnx")
    print("Training complete. Use 'runs/detect/railway_surveillance/weights/best.pt'")


def main(source=0, display=True):
    if not os.path.exists(Config.YOLO_MODEL_PATH):
        print(f"'{Config.YOLO_MODEL_PATH}' not found.")
        print("Option A: Run train_model() first to train on your dataset.")
        print("Option B: Change Config.YOLO_MODEL_PATH to an existing .pt file.")
        print("Falling back to yolov8n.pt pretrained (person/bag/weapon not custom-trained)")
        Config.YOLO_MODEL_PATH = "yolov8n.pt"

    model   = YOLO(Config.YOLO_MODEL_PATH)
    tracker = DeepSort(
        max_age             = Config.MAX_AGE,
        n_init              = Config.N_INIT,
        max_iou_distance    = Config.MAX_IOU_DISTANCE,
        max_cosine_distance = Config.MAX_COSINE_DISTANCE,
        nn_budget           = Config.NN_BUDGET,
    )
    behavior_engine = BehaviorEngine()
    alert_system    = AlertSystem()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    out_path = "output_surveillance.mp4"
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (fw, fh))

    frame_id = 0
    print(f"Processing '{source}' ...  (press Q to quit if display=True)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        timestamp = time.time()

        results    = model(frame, conf=Config.CONF_THRESHOLD, verbose=False)[0]
        detections = parse_detections(results)

        tracks = tracker.update_tracks(detections, frame=frame)
        active_tracks = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            cls  = track.det_class if track.det_class else "unknown"
            active_tracks[tid] = {
                "id": tid, "bbox": ltrb,
                "class": cls, "timestamp": timestamp
            }

        alerts = behavior_engine.analyze(active_tracks, frame, timestamp)

        for alert in alerts:
            alert_system.trigger(alert, frame, frame_id)

        vis = draw(frame, active_tracks, alerts,
                   behavior_engine, behavior_engine.bag_status)
        writer.write(vis)

        if display:
            cv2.imshow("Railway Surveillance", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_id % 100 == 0:
            print(f"Frame {frame_id} | Active tracks: {len(active_tracks)} "
                  f"| Alerts this frame: {len(alerts)}")

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Done. Output saved to '{out_path}'")
    print(f"Alert snapshots in '{Config.ALERT_OUTPUT_DIR}'")
    print("Log file: alerts.log")

    try:
        from IPython.display import HTML
        from base64 import b64encode
        with open(out_path, "rb") as f:
            video_data = b64encode(f.read()).decode()
        display_html = HTML(f"""
        <video width="900" controls autoplay loop>
          <source src="data:video/mp4;base64,{video_data}" type="video/mp4">
        </video>""")
        from IPython import display as ipydisplay
        ipydisplay.display(display_html)
    except Exception:
        pass


VIDEO_SOURCE = "mp4"   

main(source=VIDEO_SOURCE, display=False)