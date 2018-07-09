from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
from threading import Lock
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = '5A0C09318A04980C51447B7A4D868C169C13305544C7522DE1D1C04'
socketio = SocketIO(app)
thread = None
thread_lock = Lock()


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    with open("logs.txt", "r") as logfile:
        while True:
            socketio.sleep(1)
            count += 1

            line = logfile.readline()
            if line:
                socketio.emit('my_response',
                              {'data': line, 'count': count},
                              namespace='/test')


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


if __name__ == '__main__':
    socketio.run(app)
