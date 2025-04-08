# Script to detect the hand grip shape and location

import mediapipe as mp
import cv2
import numpy as np

# Hyper parameters
SMALL_GRIP_THRESHOLD = 0.18
LARGE_GRIP_THRESHOLD = 0.2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

# Function to calculate pseudo-3D Euclidean distance
def pseudo_3d_distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1) # Flip on horizontal
        
        image.flags.writeable = False
        
        results = hands.process(image) # Detections
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
                    grip = "Small"
                elif avg_dist > LARGE_GRIP_THRESHOLD:
                    grip = "Large"
                else:
                    grip = "Uncertain"

                # Display info
                cv2.putText(image, f"Trgt Obj: {grip}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Avg Dist: {avg_dist:.3f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()