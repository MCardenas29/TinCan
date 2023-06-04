from flask import Flask
from threading import Lock
from firebase import firebase


fb = firebase.FirebaseApplication("https://yolotroll-6ef4a-default-rtdb.firebaseio.com/", None)
app = Flask(__name__)
frames = [None, None]
in_lock = Lock()
