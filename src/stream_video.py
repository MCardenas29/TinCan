from flask import Response
from src.app import app, frames, in_lock
import cv2

def get_frame():
    while True:
        frame_encoded = cv2.imencode('.jpg', frames[0])[1]
        data = frame_encoded.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def stream_vid():
   return  Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
