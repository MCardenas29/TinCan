from flask import Flask
from threading import Lock

app = Flask(__name__)
frames = [None, None]
in_lock = Lock()
