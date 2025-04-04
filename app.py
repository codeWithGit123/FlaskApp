from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from bson import ObjectId
from PIL import Image
import bcrypt
import pymongo
import numpy as np
import base64
import io
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
socketio = SocketIO(app)

# Load YOLO model
model = YOLO("Pred.pt")

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://mdabdur2004:ArFeb2004@cluster0.zq6ldvu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["agroguard"]
users_collection = db["users"]
images_collection = db["images"]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def save_image(user_id, image_bytes):
    images_collection.insert_one({
        "user_id": user_id,
        "image": image_bytes
    })

def get_user_images(user_id):
    return list(images_collection.find({"user_id": user_id}))

def delete_image(image_id):
    images_collection.delete_one({"_id": ObjectId(image_id)})

def preprocess_image(image):
    image_np = np.array(image)
    resized = cv2.resize(image_np, (640, 640))
    gaussian = cv2.GaussianBlur(resized, (0, 0), 3)
    sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    brightness_factor = np.random.uniform(0.8, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return augmented

def detect_weeds(image):
    results = model(image)
    detected = False
    result_image = None
    weeds_info = []
    for result in results:
        if len(result.boxes) > 0:
            detected = True
            result_image = result.plot()
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id] if model.names and cls_id in model.names else f"class_{cls_id}"
                weeds_info.append((name, conf))
    return detected, result_image, weeds_info

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({"username": username}):
            flash("Username already exists", "warning")
        else:
            hashed_pw = hash_password(password)
            users_collection.insert_one({"username": username, "password": hashed_pw})
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username})
        if user and verify_password(password, user['password']):
            session['user'] = {"_id": str(user['_id']), "username": user['username']}
            flash("Login successful!", "success")
            return redirect(url_for('detect'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully", "success")
    return redirect(url_for('home'))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if 'user' not in session:
        return redirect(url_for('login'))

    image_data = None
    result_image = None
    weeds_info = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            image_data = encode_image(image)  # Convert uploaded image to Base64
            preprocessed = preprocess_image(image)
            detected, result_array, weeds_info = detect_weeds(preprocessed)

            if detected:
                result_image = encode_image(Image.fromarray(result_array))  # Convert result image to Base64
                save_image(session['user']['_id'], base64.b64decode(result_image))  # Save as bytes in DB
                flash("Weed detected and image saved!", "success")
            else:
                flash("No weeds detected.", "info")

    return render_template('detect.html', image=image_data, result=result_image, weeds=weeds_info)


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    images = []
    for img in get_user_images(session['user']['_id']):
        img['image'] = base64.b64encode(img['image']).decode('utf-8')  # Convert to Base64 String
        images.append(img)

    return render_template('history.html', images=images)

@app.route('/delete/<image_id>')
def delete(image_id):
    delete_image(image_id)
    flash("Image deleted successfully.", "success")
    return redirect(url_for('history'))

@app.route('/download/<image_id>')
def download(image_id):
    img_doc = images_collection.find_one({"_id": ObjectId(image_id)})
    if img_doc:
        return send_file(io.BytesIO(img_doc['image']), mimetype='image/jpeg', as_attachment=True, download_name=f"detection_{image_id}.jpg")
    return "Image not found", 404

@app.route('/camera')
def camera():
    return render_template('camera.html')

@socketio.on('video_feed')
def handle_video_feed(data):
    try:
        frame_data = data.get('frame')
        if not frame_data:
            return

        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame)
        weeds = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                weeds.append(f"{name} ({conf:.2f})")

        emit('response', {'message': 'Frame processed', 'weeds': weeds})
    except Exception as e:
        emit('response', {'message': 'Error processing frame', 'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000)
