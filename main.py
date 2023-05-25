from src.app import app
import src.inference as inference
import src.stream_video
from threading import Event, Thread

def main():
    global app
    # Inference thread arguments
    inference_event = Event()
    inference_thread = Thread(target = inference.main, args=[inference_event])
    inference_thread.start()
    # Run the flask server
    app.run(host = 'localhost', port = 8080, threaded = False)
    inference_event.set()
    del(app)

if __name__ == '__main__':
    main()
