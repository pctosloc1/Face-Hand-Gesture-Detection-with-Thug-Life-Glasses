import cv2
import numpy as np
import mediapipe as mp
import threading
from deepface import DeepFace
import time
import os

glasses_path = "thug_life_glasses.png"
if not os.path.exists(glasses_path):
    raise FileNotFoundError(f"Image file '{glasses_path}' not found! Place it in the same directory.")
glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

face_analysis_results = {}
deepface_lock = threading.Semaphore(2)
ily_detected = False

def analyze_face(face_id, face_roi):
    global face_analysis_results
    try:
        with deepface_lock:
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_roi_resized = cv2.resize(face_roi_rgb, (224, 224))

            analysis = DeepFace.analyze(face_roi_resized, actions=['age', 'gender', 'emotion'], enforce_detection=False, detector_backend='opencv')

            if analysis:
                face_analysis_results[face_id] = {
                    "age": int(analysis[0].get('age', 25)),
                    "gender": "Male" if analysis[0].get('dominant_gender', "Unknown").lower() in ["man", "male"] else "Female",
                    "emotion": analysis[0].get('dominant_emotion', "Neutral"),
                }
            else:
                face_analysis_results[face_id] = {"age": "Unknown", "gender": "Unknown", "emotion": "Unknown"}
    except Exception as e:
        print(f"DeepFace error: {e}")
        face_analysis_results[face_id] = {"age": "Unknown", "gender": "Unknown", "emotion": "Unknown"}

def detect_ily_sign(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4].y
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y
    return (index_tip < middle_tip) and (pinky_tip < ring_tip) and (thumb_tip < middle_tip)

def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    for i in range(h):
        for j in range(w):
            if y + i >= background.shape[0] or x + j >= background.shape[1]:
                continue
            alpha = overlay_resized[i, j, 3] / 255.0
            if alpha > 0:
                background[y + i, x + j] = (1 - alpha) * background[y + i, x + j] + alpha * overlay_resized[i, j, :3]

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb_frame)
    ily_detected = False
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if detect_ily_sign(hand_landmarks):
                ily_detected = True

    face_result = face_detection.process(rgb_frame)
    mesh_result = face_mesh.process(rgb_frame)
    if face_result.detections:
        for i, detection in enumerate(face_result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            if 0 <= x < iw and 0 <= y < ih and 0 < x + w <= iw and 0 < y + h <= ih:
                face_roi = frame[y:y+h, x:x+w].copy()
                if i == 0 and (time.time() - prev_time) > 1.5:
                    threading.Thread(target=analyze_face, args=(i, face_roi)).start()
                    prev_time = time.time()

                if mesh_result.multi_face_landmarks:
                    for face_landmarks in mesh_result.multi_face_landmarks:
                        left_eye = face_landmarks.landmark[33]
                        right_eye = face_landmarks.landmark[263]
                        left_x, left_y = int(left_eye.x * iw), int(left_eye.y * ih)
                        right_x, right_y = int(right_eye.x * iw), int(right_eye.y * ih)
                        glasses_w = int(abs(right_x - left_x) * 2.0)
                        glasses_h = int(glasses_w * 0.5)
                        glasses_x = left_x - int(glasses_w * 0.25)
                        glasses_y = left_y - int(glasses_h * 0.5)
                        if ily_detected:
                            overlay_image(frame, glasses, glasses_x, glasses_y, glasses_w, glasses_h)

                if i in face_analysis_results:
                    text_x = x + w + 10
                    text_y = y + 20
                    cv2.putText(frame, f"Age: {face_analysis_results[i]['age']}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"Gender: {face_analysis_results[i]['gender']}", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, face_analysis_results[i]['emotion'], (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    fps = 1 / (time.time() - prev_time + 0.001)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if ily_detected:
        cv2.putText(frame, "ANG CUTE KOOOOOOOOOO", (50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 8, 255), 3)
    cv2.imshow("WAG NA MAG CAPSTONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()