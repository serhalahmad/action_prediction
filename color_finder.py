# This code is used to find the BGR (RGB) values of a certain color using a camera. Just click on the object and it will print the values.

import cv2

cap = cv2.VideoCapture(1)

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = frame[y, x]
        print(f"Clicked color - BGR: ({b}, {g}, {r}) | RGB: ({r}, {g}, {b})")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", pick_color)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
