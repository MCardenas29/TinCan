import cv2
import torch
import time
from firebase import firebase
from datetime import datetime

firebase = firebase.FirebaseApplication("https://yolotroll-6ef4a-default-rtdb.firebaseio.com", None)

model = torch.hub.load("ultralytics/yolov5","custom","best.onnx")

init_time = time.time()
current_seg = 0
five_count = 0
minuts = 0
total_cans = 0
good_cans = 0
bad_cans = 0
good_prom = 0
bad_prom = 0
video = cv2.VideoCapture(0)
while(video.isOpened()):
	#video.set(cv2.CAP_PROP_FPS,25)
	ret,frame = video.read()
	#print(video.get(cv2.CAP_PROP_FPS))
	if ret == True:
		frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
		res = model(frame)
		res.render()
		#print(res.pandas().xyxy[0].items())
		diff = time.time() - init_time

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
			good_prom = good_cans / total_cans
			bad_prom = bad_cans / total_cans
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
			god_res = firebase.post('/buenas', datos_g)
			bad_res = firebase.post('/malas', datos_b)

			total_cans = 0
			good_cans = 0
			bad_cans = 0
			good_prom = 0
			bad_prom = 0
			minuts = 0

		cv2.imshow('Camara', res.ims[0])
		#print(res)
		#res.save(save_dir='results')
		k= cv2.waitKey(20)
		if k == 113:
			break
	else:
		break

video.release()
cv2.destroyAllWindows()