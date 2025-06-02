import cv2
from ultralytics import YOLO

# Path to your trained YOLOv8 model
model_path = 'Hasil Training/weights/best.pt'  # Change this to your model file

# Load the YOLOv8 model
model = YOLO(model_path)

# Open webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Live Inference', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()