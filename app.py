import imutils
from flask import render_template, Response, request
from src.camera.camera_opencv import Camera
from flask_socketio import SocketIO
from flask import Flask
import flask
from src.generator.VideoGenerator import VideoGenerator
import threading
from threading import Thread
import time
import numpy as np
import cv2


app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
thread = None


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@app.before_first_request
def before_first_request():
    pass


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/securityCamera')
def security_camera():
    return render_template('test.html', async_mode=socketio.async_mode)


@app.route('/security')
def security():
    return Response(VideoGenerator.get(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('mes', namespace='/send')
def test_message(message: dict):
    VideoGenerator.set(data=message.get("binary"))


if __name__ == '__main__':
    socketio.run(app=app, host='0.0.0.0', port=8000,)

''' This is 5craft http://192.168.88.25:8000'''
''' Home http://192.168.43.122:8000'''
'''ifconfig | grep "inet " | grep -v 127.0.0.1'''
'''python app.py runserver --host 0.0.0.0'''
'''  http://192.168.43.122:5000 '''
