import cv2
import dlib
from math import hypot
import numpy as np

# Hyper parameters
GAZE_THRESHOLD = 70 # Threshold to detect the gaze (related to the eye color)
LOW_CENTER_THRESHOLD = 0.5 # Threshold of the ratio to detect if we are looking right or left
HIGH_CENTER_THRESHOLD = 1.5 # Threshold of the ratio to detect if we are looking right or left

cap = cv2.VideoCapture(1)

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

def get_gaze_ratio(eye_points, facial_landmarks):
    eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32) 
    # cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, GAZE_THRESHOLD, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    right_side_threshold = threshold_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    try:
        gaze_ratio = left_side_white / right_side_white
    except:
        # this is the case when right white are zero which occurs when the right eye is not detected (blinking for example)
        return -1
    return gaze_ratio

while True:
    ret, frame = cap.read()
    color_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Gaze Detection
        gaze_ratio_right_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_left_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        
        if gaze_ratio <= LOW_CENTER_THRESHOLD:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            color_frame[:] = (0, 0, 255) # if the frame is red then we are looking right
        elif LOW_CENTER_THRESHOLD < gaze_ratio < HIGH_CENTER_THRESHOLD: 
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else: 
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            color_frame[:] = (255, 0, 0) # if the frame is blue then we are looking left

    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Color Frame', color_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()