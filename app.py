
from flask import Flask, jsonify, render_template, Response,request
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

@app.route('/asana2')
def asana2():
    return render_template('asana2.html')

@app.route('/asana3')
def asana3():
    return render_template('asana3.html')

@app.route('/asana4')
def asana4():
    return render_template('asana4.html')

@app.route('/asana5')
def asana5():
    return render_template('asana5.html')


@app.route('/asana6')
def asana6():
    return render_template('asana6.html')


@app.route('/asana7')
def asana7():
    return render_template('asana7.html')


@app.route('/asana8')
def asana8():
    return render_template('asana8.html')


@app.route('/asana9')
def asana9():
    return render_template('asana9.html')


@app.route('/asana10')
def asana10():
    return render_template('asana10.html')


@app.route('/asana11')
def asana11():
    return render_template('asana11.html')


@app.route('/asana12')
def asana12():
    return render_template('asana12.html')








latest_result = {
    "accuracy": 0,
    "hold_time": 0,
    "completed": False
}


# ---------- CONFIG ----------
SESSION_TIME = 30
MAX_HOLD_TIME = 30

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
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return angle


# ---------- GET ANGLE ----------
def get_angle_value(lm, name):
    try:
        if name == "left_elbow":
            return calculate_angle([lm[11].x, lm[11].y],
                                   [lm[13].x, lm[13].y],
                                   [lm[15].x, lm[15].y])

        elif name == "right_elbow":
            return calculate_angle([lm[12].x, lm[12].y],
                                   [lm[14].x, lm[14].y],
                                   [lm[16].x, lm[16].y])

        elif name == "left_knee":
            return calculate_angle([lm[23].x, lm[23].y],
                                   [lm[25].x, lm[25].y],
                                   [lm[27].x, lm[27].y])

        elif name == "right_knee":
            return calculate_angle([lm[24].x, lm[24].y],
                                   [lm[26].x, lm[26].y],
                                   [lm[28].x, lm[28].y])

        elif name == "left_shoulder":
            return calculate_angle([lm[13].x, lm[13].y],
                                   [lm[11].x, lm[11].y],
                                   [lm[23].x, lm[23].y])

        elif name == "right_shoulder":
            return calculate_angle([lm[14].x, lm[14].y],
                                   [lm[12].x, lm[12].y],
                                   [lm[24].x, lm[24].y])
    except:
        return None


# ---------- LIVE VIDEO STREAM ----------
def generate_frames(pose_id):

    cap = cv2.VideoCapture(0)

    # ✅ DYNAMIC FILES
    json_file = f"Pose{pose_id}-P.json"
    image_file = f"static/Pose{pose_id}-P.jpeg"
    # LOAD JSON
    with open(json_file, "r") as f:
        pose_data = json.load(f)

    pose_name = pose_data["pose_name"]
    pose_ranges = pose_data["angles"]

    # LOAD IMAGE
    ref_img = cv2.imread(image_file)
    ref_img = cv2.resize(ref_img, (200, 150))

    correct_hold_time = 0
    pose_hold_start = None
    session_start = time.time()

    total_frames = 0
    correct_frames = 0

    session_ended = False   # ✅ ADDED

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 720))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            current_time = time.time()
            elapsed = int(current_time - session_start)

            pose_correct = True
            feedback = []

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                total_frames += 1

                for joint, (min_v, max_v) in pose_ranges.items():

                    angle = get_angle_value(lm, joint)

                    if angle is None or not (min_v <= angle <= max_v):
                        pose_correct = False
                        feedback.append(f"Fix {joint}")

                        if joint in joint_highlight_map:
                            idx = joint_highlight_map[joint]
                            h, w, _ = frame.shape
                            x = int(lm[idx].x * w)
                            y = int(lm[idx].y * h)
                            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

                if pose_correct:
                    correct_frames += 1
                    if pose_hold_start is None:
                        pose_hold_start = current_time
                else:
                    if pose_hold_start:
                        correct_hold_time += current_time - pose_hold_start
                        pose_hold_start = None

            else:
                feedback.append("No person detected")

            current_hold = int(current_time - pose_hold_start) if pose_hold_start else 0

            # ✅ FIXED HOLD + STOP LOGIC
            total_hold = correct_hold_time + current_hold

            if elapsed >= SESSION_TIME or total_hold >= MAX_HOLD_TIME:
                session_ended = True

            # ---------- DISPLAY ----------
            cv2.putText(frame, pose_name, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"Time: {elapsed}/{SESSION_TIME}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Hold: {int(total_hold)}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y = 150
            for msg in feedback:
                cv2.putText(frame, msg, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25

            h, w, _ = frame.shape
            frame[10:160, w-210:w-10] = ref_img

            # ---------- STREAM FRAME ----------
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # ✅ STOP STREAM PROPERLY
            if session_ended:
                break


        # ✅ CALCULATE RESULT
        accuracy = round((correct_frames / total_frames) * 100, 2) if total_frames else 0

        latest_result["accuracy"] = accuracy
        latest_result["hold_time"] = round(total_hold, 2)
        latest_result["completed"] = True




    cap.release()

@app.route('/video')
def video():
    global latest_result

    # ✅ GET POSE ID FROM FRONTEND
    pose_id = request.args.get("pose", default=1, type=int)

    latest_result = {
        "accuracy": 0,
        "hold_time": 0,
        "completed": False
    }

    return Response(generate_frames(pose_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- RESULT API ----------
@app.route("/start_session")
def start_session():
    return jsonify({
        "message": "Session handled in video stream",
        "status": "running"
    })

@app.route("/get_result")
def get_result():
    return jsonify(latest_result)


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)