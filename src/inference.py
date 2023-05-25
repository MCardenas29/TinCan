import os
import time
import torch
import cv2
import src.app as app
from threading import Event

FPS = 30
FRAME_SIZE = 640/2, 480/2
DEV_INDEX = 0

# Thread main routine
def main(event):
    last_timestamp = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', './can_recognition.pt')
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
        current_time = time.time()
        if (current_time - last_timestamp) < (1/FPS):
            continue
        last_timestamp = current_time
        # Get the current frame
        ret, frame = camera.read()
        if not ret:
            continue
        inference = model(frame)
        inference.render()
        # Write the current frame in ~disk~ ram
        # with app.in_lock:
        app.frames[0] = frame
        app.frames[1] = inference.ims[0]
        # cv2.imwrite('current_frame.png', frame)
        # cv2.imwrite('inference_frame.png', inference.ims[0])

    camera.release()
    del(camera)
    del(model)

if __name__ == '__main__':
    main(Event())
