
import cv2
from ultralytics import YOLO
import datetime
import os


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("outputs/recorded.avi", fourcc, 20.0, (640, 480))

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_file = open("logs/detections.txt", "a")


mode = "original"

print("🔢 Press keys to change view:")
print("1 - Original, 2 - Grayscale, 3 - Edge, 4 - Blurred, 5 - Inverted")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    processed = frame.copy()

    
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

    
    results = model(processed, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(processed, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {label}: {conf:.2f} in mode: {mode}\n")

    
    out.write(processed)

    
    cv2.imshow("YOLOv8 Object Tracker", processed)

    
    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == ord('s'):  
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
