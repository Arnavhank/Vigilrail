from ultralytics import YOLO
import cv2


MODEL_PATH = "best2.pt"
VIDEO_PATH = "video9.mp4"

model = YOLO(MODEL_PATH)
print("Classes:", model.names)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 5 != 0:
        continue

    results = model(frame, conf=0.25, verbose=False)[0]

    unattended = 0
    attended   = 0
    persons    = 0

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])].lower()

        if "unattended" in cls_name:
            unattended += 1
        elif "attended" in cls_name:
            attended += 1
        elif "person" in cls_name:
            persons += 1

    # ── Draw Boxes ──────────────────────────────────────────
    img = results.plot()
    h, w = img.shape[:2]

    # ── Alert Banner ────────────────────────────────────────
    if unattended > 0:
        cv2.rectangle(img, (0, 0), (w, 65), (0, 0, 200), -1)
        cv2.putText(img,
            f"ALERT: {unattended} UNATTENDED BAG(S)",
            (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9,
            (255, 255, 255), 2)
    else:
        cv2.rectangle(img, (0, 0), (w, 65), (0, 150, 0), -1)
        cv2.putText(img,
            "SAFE - No Unattended Objects",
            (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9,
            (255, 255, 255), 2)


    cv2.rectangle(img, (0, h-40), (w, h), (30, 30, 30), -1)
    cv2.putText(img,
        f"Frame: {frame_count} | Persons: {persons} | Attended: {attended} | Unattended: {unattended}",
        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1)


    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Processing Done!")