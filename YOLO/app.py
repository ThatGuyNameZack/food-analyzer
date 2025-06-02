from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your YOLOv8 model
model = YOLO('Hasil Training/weights/best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'

        # Save the uploaded image
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Read and run YOLOv8 inference
        frame = cv2.imread(filepath)
        results = model(frame)
        annotated_frame = results[0].plot()

        # Save the annotated image
        result_path = os.path.join(UPLOAD_FOLDER, f"result_{unique_filename}")
        cv2.imwrite(result_path, annotated_frame)

        return render_template('index.html', result_image=result_path)

    return render_template('index.html', result_image=None)

if __name__ == '__main__':
    app.run(debug=True)
