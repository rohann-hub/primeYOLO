# from ultralytics import YOLO
# import cv2, cvzone

# model = YOLO("yolov8n-oiv7.pt")
# classNames = model.names

# cap = cv2.VideoCapture(0)
# cap.set(3,1280); cap.set(4,720)

# while True:
#     _, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         for box in r.boxes:
#             x1,y1,x2,y2 = map(int, box.xyxy[0])
#             cvzone.cornerRect(img, (x1,y1,x2-x1,y2-y1), l=9, rt=3)
#             conf = round(float(box.conf[0])*100,2)
#             cls = int(box.cls[0])
#             label = classNames.get(cls, f"ID{cls}")
#             cvzone.putTextRect(img, f"{label} {conf}%", (x1, max(35,y1)), scale=0.7, thickness=1)
#     cv2.imshow("OIV7 Detect", img)
#     if cv2.waitKey(1)==ord('q'): break

# cap.release()
# cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO
import datetime
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("outputs/recorded.avi", fourcc, 20.0, (640, 480))

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_file = open("logs/detections.txt", "a")

# Default mode
mode = "original"

print("🔢 Press keys to change view:")
print("1 - Original, 2 - Grayscale, 3 - Edge, 4 - Blurred, 5 - Inverted")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    processed = frame.copy()

    # Apply mode
    if mode == "gray":
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # for YOLO overlay
    elif mode == "edge":
        edges = cv2.Canny(frame, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == "blur":
        processed = cv2.GaussianBlur(frame, (15, 15), 0)
    elif mode == "invert":
        processed = cv2.bitwise_not(frame)

    # Run YOLO detection
    results = model(processed, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw boxes on processed image
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(processed, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Log detections
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {label}: {conf:.2f} in mode: {mode}\n")

    # Save the frame
    out.write(processed)

    # Show the result
    cv2.imshow("YOLOv8 Object Tracker", processed)

    # Key controls
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):  # Snapshot
        snap = f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join("outputs", snap), processed)
        print(f"📸 Snapshot saved: {snap}")
    elif key == ord('1'):
        mode = "original"
        print("🎨 Mode: Original")
    elif key == ord('2'):
        mode = "gray"
        print("🎨 Mode: Grayscale")
    elif key == ord('3'):
        mode = "edge"
        print("🎨 Mode: Edge")
    elif key == ord('4'):
        mode = "blur"
        print("🎨 Mode: Blur")
    elif key == ord('5'):
        mode = "invert"
        print("🎨 Mode: Inverted")

log_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()