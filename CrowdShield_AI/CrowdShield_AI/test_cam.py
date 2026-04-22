import cv2
import sys
print("OpenCV version:", cv2.__version__)
cap = cv2.VideoCapture(0)
is_open = cap.isOpened()
print("cap.isOpened():", is_open)
if is_open:
    ret, frame = cap.read()
    print("cap.read() returned:", ret)
else:
    print("Could not open camera 0.")
cap.release()
