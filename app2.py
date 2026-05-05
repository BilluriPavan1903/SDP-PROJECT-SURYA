from flask import Flask, jsonify, render_template, Response, request
from flask_cors import CORS

import cv2
import mediapipe as mp
import numpy as np
import time
import json

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose

# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/asana1')
def asana1():
    return render_template('asana1.html')

# ---------- RESULT STORAGE ----------
latest_result = {}

# ---------- CONFIG ----------
SESSION_TIME = 100

joint_highlight_map = {
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
}

# ---------- ANGLE FUNCTION ----------
def calculate_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ---------- GET ANGLE ----------
def get_angle_value(lm, name):
    try:
        if name == "left_elbow":
            return calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
        elif name == "right_elbow":
            return calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
        elif name == "left_knee":
            return calculate_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
        elif name == "right_knee":
            return calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
        elif name == "left_shoulder":
            return calculate_angle([lm[13].x, lm[13].y], [lm[11].x, lm[11].y], [lm[23].x, lm[23].y])
        elif name == "right_shoulder":
            return calculate_angle([lm[14].x, lm[14].y], [lm[12].x, lm[12].y], [lm[24].x, lm[24].y])
    except:
        return None

# ---------- MAIN STREAM ----------
def generate_frames(pose_id):
    global latest_result   # ✅ IMPORTANT

    cap = cv2.VideoCapture(0)

    with open(f"Pose{pose_id}-P.json", "r") as f:
        pose_data = json.load(f)

    joint_configs = pose_data["joints"]
    MAX_HOLD_TIME = pose_data.get("target_hold_time", 20)

    correct_hold_time = 0
    pose_hold_start = None
    session_start = time.time()

    with mp_pose.Pose(0.5, 0.5) as pose:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 720))
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_time = time.time()

            feedback = []
            deviation_map = {}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                total_score = 0
                total_weight = 0

                left_angles, right_angles = [], []

                for joint, config in joint_configs.items():

                    angle = get_angle_value(lm, joint)
                    if angle is None:
                        continue

                    min_v, max_v = config["min"], config["max"]
                    ideal, weight = config["ideal"], config["weight"]

                    deviation = abs(angle - ideal)
                    deviation_map[joint] = round(deviation, 2)

                    tolerance = (max_v - min_v) / 2
                    score = max(0, 1 - (deviation / tolerance))

                    total_score += score * weight
                    total_weight += weight

                    if not (min_v <= angle <= max_v):
                        feedback.append(config["feedback"])

                    if "left" in joint:
                        left_angles.append(angle)
                    else:
                        right_angles.append(angle)

                pose_score = round((total_score / total_weight) * 100, 2) if total_weight else 0

                balance = 100 - abs(np.mean(left_angles) - np.mean(right_angles)) if left_angles and right_angles else 0
                stability = 100 if pose_score > 70 else 60
                confidence = 0.9

                hold_quality = "Excellent" if pose_score > 85 else "Good" if pose_score > 65 else "Needs Improvement"

            else:
                pose_score = 0
                balance = 0
                stability = 0
                confidence = 0
                hold_quality = "No Detection"

            total_hold = correct_hold_time

            # ---------- FINAL RESULT (UPDATED CORRECTLY) ----------
            latest_result = {
                "pose_score": pose_score,
                "stability": stability,
                "balance": balance,
                "hold_time": total_hold,
                "hold_quality": hold_quality,
                "confidence": confidence,
                "deviation": deviation_map,
                "feedback": feedback,
                "completed": True
            }

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# ---------- ROUTES ----------
@app.route('/video')
def video():
    global latest_result

    pose_id = request.args.get("pose", default=1, type=int)

    latest_result = {"completed": False}  # ✅ FIXED

    return Response(generate_frames(pose_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_result")
def get_result():
    return jsonify(latest_result)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)