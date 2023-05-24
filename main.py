from flask import Flask, Response
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'can_recognition.pt')
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    del(camera)
    exit

app = Flask(__name__)

def frame():
    retval, img = camera.read()
    imencode = cv2.imencode('.jpeg', img)[1]
    data = imencode.tobytes()
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'

@app.route('/')
def entry():
    return Response(frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host = 'localhost', port = '8080', threaded = True)
    camera.release()
    del(camera)
    del(app)

