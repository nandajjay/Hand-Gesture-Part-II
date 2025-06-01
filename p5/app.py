from flask import Flask, render_template, Response, jsonify
from gesture_control import GestureController
from object_detection import ObjectDetector
import cv2
import numpy as np  # ADD THIS

app = Flask(__name__)
controller = GestureController()
detector = ObjectDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        frame_bytes = controller.get_frame()  # JPEG bytes
        if frame_bytes is not None:
            # Decode bytes to OpenCV image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run object detection and draw boxes
            detections = detector.detect_objects(frame)
            frame = detector.draw_detections(frame, detections)

            # Re-encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_with_detections = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_with_detections + b'\r\n')

@app.route('/status')
def get_status():
    status = controller.get_status()
    status['detections'] = controller.get_detections()
    return jsonify(status)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        controller.stop()
