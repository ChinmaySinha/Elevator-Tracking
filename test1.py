from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model1.h5')

app = Flask(__name__)

# Variables for prediction smoothing
last_prediction = None
stable_prediction = None
stable_count = 0
threshold = 3  # Number of frames to wait for a stable prediction

def preprocess_frame(frame):
    # Crop the center region to focus on the digit
    height, width, _ = frame.shape
    crop_size = min(height, width)
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2
    frame_cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Convert to grayscale and resize to 256x256
    frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (256, 256))

    # Normalize and add batch dimension
    frame_normalized = frame_resized / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=(0, -1))  # Shape becomes (1, 256, 256, 1)
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
            # Predict the current frame's digit
            predicted_digit = predict_frame(frame)

            # Display the prediction on the frame
            cv2.putText(frame, f"Predicted Number: {predicted_digit}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
