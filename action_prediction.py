# Script to detect the hand grip shape and location

import mediapipe as mp
import cv2
import dlib
from math import hypot
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Hyper parameters
SMALL_GRIP_THRESHOLD = 0.18
LARGE_GRIP_THRESHOLD = 0.2
GAZE_THRESHOLD = 70 # Threshold to detect the gaze (related to the eye color)
LOW_CENTER_THRESHOLD = 0.5 # Threshold of the ratio to detect if we are looking right or left
HIGH_CENTER_THRESHOLD = 1.5 # Threshold of the ratio to detect if we are looking right or left
# Hardcoded object centers (in pixel coordinates)
# Example: (x, y)
small_object_center = (200, 300)
large_object_center = (400, 250)

### Initializing the Bayesian Network (BN)
# 1. Define the model structure
model = DiscreteBayesianNetwork([
    ('GazeDirection', 'TargetObject'),
    ('GripShape', 'TargetObject'),
    ('HandProximity', 'TargetObject'),
    ('Hand3DCloseness', 'TargetObject')
])

# 2. Define state names for all nodes
state_names = {
    'GazeDirection': ['towards_small', 'towards_large'],
    'GripShape': ['small', 'large'],
    'HandProximity': ['near_small', 'near_large'],
    'Hand3DCloseness': ['close', 'far'],
    'TargetObject': ['small', 'large']
}

# 3. Define CPDs for input nodes (uniform for now)
cpd_gaze = TabularCPD('GazeDirection', 2, [[0.5], [0.5]], state_names=state_names)
cpd_grip = TabularCPD('GripShape', 2, [[0.5], [0.5]], state_names=state_names)
cpd_proximity = TabularCPD('HandProximity', 2, [[0.5], [0.5]], state_names=state_names)
cpd_closeness = TabularCPD('Hand3DCloseness', 2, [[0.5], [0.5]], state_names=state_names)  # Now only close and far

# 4. Define full CPD for TargetObject (manually filled)
cpd_values = [
    [0.99, 0.99, 0.3, 0.95, 0.85, 0.95, 0.05, 0.95, 0.95, 0.05, 0.15, 0.05, 0.95, 0.05, 0.01, 0.01],  # Probabilities for "small"
    [0.01, 0.01, 0.7, 0.05, 0.15, 0.05, 0.95, 0.05, 0.05, 0.95, 0.85, 0.95, 0.05, 0.95, 0.99, 0.99]   # Probabilities for "large"
]

# Create the full CPD for TargetObject
cpd_target = TabularCPD(
    variable='TargetObject', variable_card=2,
    values=cpd_values,
    evidence=['GazeDirection', 'GripShape', 'HandProximity', 'Hand3DCloseness'],
    evidence_card=[2, 2, 2, 2],
    state_names=state_names
)

# 5. Add all CPDs to the model
model.add_cpds(cpd_gaze, cpd_grip, cpd_proximity, cpd_closeness, cpd_target)

# 6. Validate the model
assert model.check_model()

# 7. Run inference
infer = VariableElimination(model)
### END of setting up the BN 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(1)

# Function to calculate pseudo-3D Euclidean distance
def pseudo_3d_distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

# Compute distances to objects
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_gaze_ratio(gray_image, eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
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
    eye = cv2.bitwise_and(gray_image, gray_image, mask=mask)

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

grip_shape = "uncertain"
gaze_direction = "uncertain"
hand_proximity = "uncertain"
hand_closeness = "uncertain"

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        color_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1) # Flip on horizontal
        
        image.flags.writeable = False
        
        results = hands.process(image) # Detections
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hand_center = None

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                                          )
                
                # Extract fingertips
                thumb_tip = hand.landmark[4]
                tip_ids = [8, 12, 16, 20] # tip landmarks of index, middle, ring, and pinky fingers
                distances = []
                # Compute distances
                for i in tip_ids:
                    fingertip = hand.landmark[i]
                    dist = pseudo_3d_distance(thumb_tip, fingertip)
                    distances.append(dist)
                # Average grip distance
                avg_dist = np.mean(distances)

                # Grip logic (you can adjust threshold based on your setup)
                if avg_dist < SMALL_GRIP_THRESHOLD:
                    grip_shape = "small"
                elif avg_dist > LARGE_GRIP_THRESHOLD:
                    grip_shape = "large"
                else:
                    grip_shape = "uncertain"

                # Display info
                cv2.putText(image, f"Trgt Obj: {grip_shape}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Avg Dist: {avg_dist:.3f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                
                # Detect hand location
                # Use average of some palm landmarks as hand center
                palm_ids = [0, 1, 5, 9, 13, 17]
                h, w, _ = image.shape
                xs = [hand.landmark[i].x * w for i in palm_ids]
                ys = [hand.landmark[i].y * h for i in palm_ids]
                hand_center = (int(np.mean(xs)), int(np.mean(ys)))

                # Draw hand center
                cv2.circle(image, hand_center, 5, (0, 255, 255), -1)

                dist_small = euclidean_dist(hand_center, small_object_center)
                dist_large = euclidean_dist(hand_center, large_object_center)

                # Decide the hand proximity
                hand_proximity = "near_small" if dist_small < dist_large else "near_large"

                # Display result
                cv2.putText(image, f"Target: {hand_proximity}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Gaze Detection
            gaze_ratio_right_eye = get_gaze_ratio(gray, [36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_left_eye = get_gaze_ratio(gray, [42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            
            if gaze_ratio <= LOW_CENTER_THRESHOLD:
                gaze_direction = "towards_large"
                color_frame[:] = (0, 0, 255) # if the frame is red then we are looking right
            elif LOW_CENTER_THRESHOLD < gaze_ratio < HIGH_CENTER_THRESHOLD: 
                gaze_direction = "uncertain"
            else: 
                gaze_direction = "towards_small"
                color_frame[:] = (255, 0, 0) # if the frame is blue then we are looking left

        # Predict the target object using BN
        query = infer.query(
            variables=['TargetObject'],
            evidence={
                'GazeDirection': gaze_direction if gaze_direction != "uncertain" else "towards_small",
                'GripShape': grip_shape if grip_shape != "uncertain" else "small",
                'HandProximity': hand_proximity if hand_proximity != "uncertain" else "near_small",
                'Hand3DCloseness': hand_closeness if hand_closeness != "uncertain" else "close"
            }
        )
        cv2.putText(image, f"P = {query.values[0]}", (200, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"P = {query.values[1]}", (400, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", image)
        cv2.imshow('Color Frame', color_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()