from flask import Flask,render_template
from flask_socketio import SocketIO,emit

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_feed')
def handle_video_feed(data):
    print("Received video feed data")
    emit('response',{'message':'Video feed received!'},broadcast=True)

if __name__ == '__main__':   
    socketio.run(app,host='0.0.0.0',port=5000)