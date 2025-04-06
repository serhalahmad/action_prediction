import cv2
import dlib
from math import hypot
import numpy as np

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    top_point = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom_point = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, top_point, bottom_point, (0, 255, 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((top_point[0] - bottom_point[0]), (top_point[1] - bottom_point[1]))
    
    return hor_line_length / ver_line_length

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Blinking Detection
        # right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        # left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        # if left_eye_ratio > 5:
        #     cv2.putText(frame, "LEFT", (50, 150), font, 3, (255, 0, 0))
        # if right_eye_ratio > 5:
        #     cv2.putText(frame, "RIGHT", (100, 300), font, 3, (255, 0, 0))
        
        # Gaze Detection
        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32) 
        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])

        right_eye = frame[min_y: max_y, min_x: max_x]
        right_eye = cv2.resize(right_eye, None, fx=5, fy=5)

        cv2.imshow("Eye", right_eye)

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()