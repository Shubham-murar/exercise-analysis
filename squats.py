#squat only 

from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
import shutil
import os

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def check_squat_feedback(landmarks):
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    avg_angle = (left_angle + right_angle) / 2
    
    if avg_angle < 80:
        feedback = "Too Low! Raise Your Hips"
    elif 80 <= avg_angle <= 110:
        feedback = "Perfect Squat!"
    elif 110 < avg_angle <= 140:
        feedback = "Almost There! Go Lower"
    else:
        feedback = "Too High! Lower Your Hips"
    
    accuracy = max(0, min(100, (1 - abs(avg_angle - 95) / 50) * 100))
    return feedback, int(accuracy)

def analyze_and_correct_squats(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP.FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            feedback, accuracy = check_squat_feedback(landmarks)
            cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        
        out.write(image)
    
    cap.release()
    out.release()
    return output_path

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    input_path = f"videos/{file.filename}"
    output_path = f"processed_videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)
    os.makedirs("processed_videos", exist_ok=True)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processed_video_path = analyze_and_correct_squats(input_path, output_path)
    return {"message": "Video processed successfully", "processed_video": processed_video_path}

