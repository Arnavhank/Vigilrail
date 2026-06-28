import cv2
import numpy as np
from ultralytics import YOLO
import os

print("Current directory:", os.getcwd())
print("Files:", os.listdir())

# Load model
model = YOLO("best.pt")

# Load video
cap = cv2.VideoCapture("/Users/sunilkumarsingh/Desktop/VIGILRAIL/tracktestvideo.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── TRAPEZIUM ZONE (CORRECTED) ─────────────────────────

        # Bottom (near camera) → wide
    bottom_left  = int(w * 0.00)
    bottom_right = int(w * 0.95)   # slightly extend right

    # Top (far distance) → wide enough to cover both tracks
    top_left  = int(w * 0.05)      # include left track
    top_right = int(w * 0.85)      # include right track (near platform edge)

    # Height (adjust if needed)
    top_cut = int(h * 0.38)

    zone_poly = np.array([
        [top_left,  top_cut],
        [top_right, top_cut],
        [bottom_right, h],
        [bottom_left,  h],
    ], np.int32)

    # ───────────────────────────────────────────────────────

    # Detection
    results = model(frame, conf=0.4, verbose=False)[0]

    # Draw zone
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_poly], (0, 0, 180))
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.polylines(frame, [zone_poly], True, (0, 255, 255), 2)

    cv2.putText(frame, "RESTRICTED ZONE",
                (top_left + 10, top_cut - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    intrusion_count = 0

    # Process detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        foot_x = (x1 + x2) // 2
        foot_y = y2

        in_zone = cv2.pointPolygonTest(
            zone_poly,
            (float(foot_x), float(foot_y)),
            False
        ) >= 0

        if in_zone:
            intrusion_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"!! ON TRACK {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0), 1)

    # Alert banner
    if intrusion_count > 0:
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
        cv2.putText(frame,
                    f"ALERT: {intrusion_count} PERSON ON TRACK!",
                    (10, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (255, 255, 255), 2)

        print(f"ALERT: {intrusion_count} person(s) on track!")

    # Show output
    cv2.imshow("VIGILRAIL", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()