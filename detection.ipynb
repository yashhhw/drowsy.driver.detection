import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import threading

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate mouth aspect ratio
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Function to process video frames
def process_frame(frame):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.6

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    eye_flag = mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth = np.array(landmarks['bottom_lip'])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            eye_flag = True
        if mar > MOUTH_AR_THRESH:
            mouth_flag = True

    return eye_flag, mouth_flag

# Function to capture video
def capture_video():
    video_cap = cv2.VideoCapture(0)
    count = score = 0

    while True:
        success, image = video_cap.read()
        if not success:
            break

        image = cv2.resize(image, (800, 500))
        count += 1

        n = 5
        if count % n == 0:
            eye_flag, mouth_flag = process_frame(image)
            score += 1 if eye_flag or mouth_flag else -1
            score = max(0, score)

        cv2.putText(image, f"Score: {score}", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if score >= 5:
            cv2.putText(image, "Drowsy", (image.shape[1] - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Drowsiness Detection', image)

        if cv2.waitKey(1) & 0xFF != 255:
            break

    video_cap.release()
    cv2.destroyAllWindows()

# Start video capture in a separate thread
video_thread = threading.Thread(target=capture_video)
video_thread.start()
