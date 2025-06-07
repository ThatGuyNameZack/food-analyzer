from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid
from nutrition_data import nutrition

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('train64/weights/best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detected_items = []
    nutrition_results = []

    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'

        # Save uploaded file
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run YOLO on image
        image = cv2.imread(filepath)
        results = model(image)[0]
        annotated = results.plot()

        # Save annotated image
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated)

        # Get detected class names
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            if label not in detected_items:
                detected_items.append(label)

        # Match with nutrition data
        for item in detected_items:
            if item in nutrition:
                nutrition_results.append({
                    'name': item,
                    **nutrition[item]
                })
            else:
                nutrition_results.append({'name': item, 'note': 'No data available'})

        result_image = result_filename

    return render_template('index.html',
                           result_image=result_image,
                           nutrition_list=nutrition_results)

if __name__ == '__main__':
    app.run(debug=True)
