import os
import time
import torch
import cv2
import src.app as app
from threading import Event
from datetime import datetime

FPS = 30
FRAME_SIZE = 640, 480
DEV_INDEX = 0

# Thread main routine
def main(event):
    init_time = time.time()
    current_seg = 0
    five_count = 0
    minuts = 0
    total_cans = 0
    good_cans = 0
    bad_cans = 0
    good_prom = 0
    bad_prom = 0
    # last_timestamp = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', './models/a.onnx')
    camera = cv2.VideoCapture(DEV_INDEX)
    # Verify that the video device is succesfully opened
    if not camera.isOpened():
        print("Error opening the camera!")
        exit
    # Configure the camera device
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    # Configure torch
    torch.no_grad()
    # Process camera
    while not event.is_set():
        # Process the framerate
        # current_time = time.time()
        # if (current_time - last_timestamp) < (1/FPS):
        #     continue
        # last_timestamp = current_time
        # Get the current frame
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 640))
        diff = time.time() - init_time
        res = model(frame)
        res.render()
        # Write the current frame in ~disk~ ram
        # with app.in_lock:
        app.frames[0] = frame
        app.frames[1] = res.ims[0]
        # cv2.imwrite('current_frame.png', frame)
        # cv2.imwrite('res.png', res.ims[0])
        if(diff >= 1):
            init_time = time.time() - diff + 1
            current_seg += 1
            five_count += 1
            for v in res.pandas().xyxy[0].get('name'):
                total_cans += 1
                if(v == 'OK'):
                    good_cans += 1
                elif(v == 'BAD'):
                    bad_cans +=1

        if(current_seg >= 60):
            minuts += 1
            current_seg = 0

        if(minuts >= 1):
            now = datetime.now()
            good_prom = (good_cans / total_cans) * 100
            bad_prom = (bad_cans / total_cans) * 100
            print(good_prom)
            print(bad_prom)
            datos_g={
                'promedio' : good_prom,
                'fecha' : now
            }
            datos_b={
                'promedio' : bad_prom,
                'fecha' : now
            }
            god_res = app.fb.post('/buenas', datos_g)
            bad_res = app.fb.post('/malas', datos_b)

            total_cans = 0
            good_cans = 0
            bad_cans = 0
            good_prom = 0
            bad_prom = 0
            minuts = 0


    camera.release()
    del(camera)
    del(model)

if __name__ == '__main__':
    main(Event())
