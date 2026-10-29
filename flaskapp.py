import os
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from pole_detector import PoleModel
import cv2

app = Flask(__name__)
model = PoleModel()

file = "temp.jpg"
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get("image")
        if not file: return "No file uploaded", 400
        file.save("temp.jpg")
    return render_template('index.html')

@app.route('/display')
def display():
    return Response(process(), mimetype="multipart/x-mixed-replace; boundary=frame")

def process():
    if os.path.exists(file):
        for result in model.pipeline_stream(file):
            frame = result["image"]
            suc, buf = cv2.imencode(".jpg", frame)
            if not suc: continue
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app.run(debug=False)