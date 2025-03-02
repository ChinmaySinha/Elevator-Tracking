from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model1.h5')

app = Flask(__name__)

# Variables for smoothing predictions
last_prediction = None
stable_prediction = None
stable_count = 0
threshold = 3  # Number of frames to wait for a stable prediction

# Authentication details
CORRECT_PASSWORD = "12345"
EMAIL_SUFFIX = "@vitstudent.ac.in"

def preprocess_frame(frame):
    height, width, _ = frame.shape
    crop_size = min(height, width)
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2
    frame_cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (256, 256))
    frame_normalized = frame_resized / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=(0, -1))  # Add batch and channel dimensions
    return frame_batch

def predict_frame(frame):
    global last_prediction, stable_prediction, stable_count
    frame_batch = preprocess_frame(frame)
    prediction = model.predict(frame_batch)
    predicted_label = np.argmax(prediction)

    # Smoothing logic for stable prediction
    if predicted_label == last_prediction:
        stable_count += 1
    else:
        stable_count = 0
        last_prediction = predicted_label

    if stable_count >= threshold:
        stable_prediction = predicted_label

    return stable_prediction

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            predicted_digit = predict_frame(frame)

            # Display the continuous prediction on the frame
            cv2.putText(frame, f"Predicted Number: {predicted_digit}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email.endswith(EMAIL_SUFFIX) and password == CORRECT_PASSWORD:
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
