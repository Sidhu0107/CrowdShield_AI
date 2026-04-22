import cv2
import threading

def run():
    cap = cv2.VideoCapture(0)
    print("In thread, cap.isOpened():", cap.isOpened())
    cap.release()

t = threading.Thread(target=run)
t.start()
t.join()
