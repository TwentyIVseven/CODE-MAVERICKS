from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

KNOWN_FACES_DIR = "faces"
EXCEL_FILE = "attendance.xlsx"

# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]

            known_encodings.append(encoding)
            known_names.append(os.path.splitext(filename)[0])  # Remove file extension

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# Video Capture
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = np.argmax(matches)
                    name = known_names[match_index]

                    # Mark attendance in Excel
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")

                    df = pd.read_excel(EXCEL_FILE)

                    if not ((df["Name"] == name) & (df["Date"] == date)).any():
                        new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
                        df = pd.concat([df, new_entry], ignore_index=True)
                        df.to_excel(EXCEL_FILE, index=False)

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
