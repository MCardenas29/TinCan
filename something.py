import threading as th
from flask import Flask, Response
import torch
import cv2

img = None
result = None
model = torch.hub.load('ultralytics/yolov5', 'custom', './can_recognition.pt')
camera = cv2.VideoCapture(0)
app = Flask(__name__)

def process(e, camera, model):
    global img
    global result
    while not e.is_set():
        retval, img = camera.read()
        if retval:
            result = model(img)


def frame():
    global img
    global result
    while True:
        if not result:
            continue
        df = result.pandas()
        imencode = cv2.imencode('.jpg', img)[1]
        data = imencode.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def entry():
    return Response(frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    ev = th.Event()
    t1 = th.Thread(target = process, args = (ev, camera, model))
    t1.start()
    app.run(host = 'localhost', port = '8080', threaded = False)
    ev.set()
    camera.release()
    del(app)

