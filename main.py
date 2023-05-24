from flask import Flask, Response
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'can_recognition.pt')
camera = cv2.VideoCapture(0)

app = Flask(__name__)

def frame():
    while True:
        retval, img = camera.read()
        if not retval:
            continue
        imencode = cv2.imencode('.jpg', img)[1]
        data = imencode.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def entry():
    return Response(frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host = 'localhost', port = '8080', threaded = True)
    camera.release()
    del(camera)
    del(app)

