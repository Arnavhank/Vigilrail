from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture("tracktestvideo.mp4")  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    frame = results[0].plot()

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()