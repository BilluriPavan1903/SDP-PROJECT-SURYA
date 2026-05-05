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









latest_result = {}


# ---------- CONFIG ----------
SESSION_TIME = 10


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






def generate_frames(pose_id):

    global latest_result

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    json_file = f"Pose{pose_id}-P.json"
    image_file = f"static/Pose{pose_id}-P.jpeg"

    with open(json_file, "r") as f:
        pose_data = json.load(f)

    pose_name = pose_data["pose_name"]
    joint_configs = pose_data["joints"]
    MAX_HOLD_TIME = pose_data.get("target_hold_time", 20)

    ref_img = cv2.imread(image_file)
    ref_img = cv2.resize(ref_img, (200, 150))

    correct_hold_time = 0
    pose_hold_start = None
    session_start = time.time()
    session_ended = False

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

            feedback = []
            deviation_map = {}

            pose_score = 0
            balance = 0
            stability = 0
            confidence = 0
            hold_quality = "No Detection"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                total_score = 0
                total_weight = 0

                left_angles, right_angles = [], []

                # ✅ STRICT OLD LOGIC
                pose_correct = True

                for joint, config in joint_configs.items():

                    angle = get_angle_value(lm, joint)
                    if angle is None:
                        continue

                    min_v = config["min"]
                    max_v = config["max"]
                    ideal = config["ideal"]
                    weight = config["weight"]

                    # 🔥 STRICT CHECK FOR HOLD
                    if not (min_v <= angle <= max_v):
                        pose_correct = False
                        feedback.append(config["feedback"])

                        if joint in joint_highlight_map:
                            idx = joint_highlight_map[joint]
                            h, w, _ = frame.shape
                            x = int(lm[idx].x * w)
                            y = int(lm[idx].y * h)
                            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

                    # 🔥 SCORING (independent of hold)
                    deviation = abs(angle - ideal)
                    deviation_map[joint] = round(deviation, 2)

                    tolerance = (max_v - min_v) / 2
                    joint_score = max(0, 1 - (deviation / tolerance))

                    total_score += joint_score * weight
                    total_weight += weight

                    if "left" in joint:
                        left_angles.append(angle)
                    else:
                        right_angles.append(angle)

                # ---------- SCORE METRICS ----------
                pose_score = round((total_score / total_weight) * 100, 2) if total_weight else 0

                if left_angles and right_angles:
                    balance = 100 - abs(np.mean(left_angles) - np.mean(right_angles))
                    balance = round(max(0, balance), 2)

                stability = 100 if pose_score > 70 else 60
                confidence = 0.9

                if pose_score > 85:
                    hold_quality = "Excellent"
                elif pose_score > 65:
                    hold_quality = "Good"
                else:
                    hold_quality = "Needs Improvement"

                # 🔥🔥 OLD HOLD LOGIC (ONLY STRICT) 🔥🔥

                if pose_correct:
                    if pose_hold_start is None:
                        pose_hold_start = current_time
                else:
                    if pose_hold_start is not None:
                        correct_hold_time += current_time - pose_hold_start
                        pose_hold_start = None

            else:
                feedback.append("No person detected")

            # 🔥 HOLD CALCULATION (FIXED)
            current_hold = (current_time - pose_hold_start) if pose_hold_start else 0
            total_hold = correct_hold_time + current_hold

            # 🔥 FINAL HOLD FIX
            if elapsed >= SESSION_TIME or total_hold >= MAX_HOLD_TIME:
                if pose_hold_start is not None:
                    correct_hold_time += current_time - pose_hold_start
                    pose_hold_start = None
                session_ended = True

            # ---------- DISPLAY ----------

            remaining_time = max(0, SESSION_TIME - elapsed)



            cv2.putText(frame, pose_name, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            

            # ⏱️ TIMER (NEW)
            cv2.putText(frame, f"Time Left: {remaining_time}s",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            #cv2.putText(frame, f"Score: {int(pose_score)}",
                       # (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Hold: {int(total_hold)}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            

            y = 150
            for msg in feedback:
                cv2.putText(frame, msg, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25

            h, w, _ = frame.shape
            frame[10:160, w-210:w-10] = ref_img

            # ---------- STREAM ----------
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if session_ended:
                break

        # ---------- FINAL RESULT ----------
        latest_result = {
            "pose_score": pose_score,
            "stability": stability,
            "balance": balance,
            "hold_time": round(total_hold, 2),
            "hold_quality": hold_quality,
            "confidence": confidence,
            "deviation": deviation_map,
            "feedback": feedback,
            "completed": True
        }

    cap.release()




def reset_result():
    """Shared helper — always resets latest_result to a clean state."""
    global latest_result
    latest_result = {
        "pose_score": 0,
        "stability": 0,
        "balance": 0,
        "hold_time": 0,
        "hold_quality": "",
        "confidence": 0,
        "deviation": {},
        "feedback": [],
        "completed": False   # ← key flag: frontend polls until this is True
    }


@app.route('/reset')
def reset():
    """Frontend calls this BEFORE starting a new session so stale
    completed:True from the previous session is wiped immediately."""
    reset_result()
    response = jsonify({"reset": True})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route('/video')
def video():
    # GET POSE ID FROM FRONTEND
    pose_id = request.args.get("pose", default=1, type=int)

    # Always reset before a new stream so /get_result
    # can never return the previous session's completed:True
    reset_result()

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
    response = jsonify(latest_result)

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)