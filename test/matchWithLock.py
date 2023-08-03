import cv2 as cv
import threading
from collections import deque
import time


class readCamera(threading.Thread):
    def __init__(self, src: int, lock):
        super(readCamera).__init__()
        self.FPS = 30
        self.src = src
        self.q = deque(maxlen=self.FPS)
        self.lock = lock
        threading.Thread.__init__(self)

    def run(self):
        cap = cv.VideoCapture(self.src)
        if not cap.isOpened():
            print("[Exception]: Error accessing camera_%d" % self.src)
            exit(0)
        while cap.isOpened():
            _, frame = cap.read()
            cv.imshow('camera_%d' % self.src, frame)
            if cv.waitKey(1) == 27:
                break
            if self.lock.acquire():
                sysTim = time.time()
                self.q.append([frame.tolist(), sysTim])
                self.lock.release()
        cv.destroyWindow("camera_%d" % self.src)
        cap.release()

class matchCamera(threading.Thread):
    def __init__(self, q_1, q_2, lock_1, lock_2):
        super(matchCamera).__init__()
        threading.Thread.__init__(self)
        self.q_1 = q_1
        self.q_2 = q_2
        self.lock_1 = lock_1
        self.lock_2 = lock_2

    def run(self):
        deltaT = 0
        while True:
            if self.lock_1.acquire() and len(self.q_1) > 0 and self.lock_2.acquire() and len(self.q_2) > 0:
                tempFrame_1 = self.q_1.pop()
                self.lock_1.release()
                tempFrame_2 = self.q_2.pop()
                self.lock_2.release()
                deltaT = tempFrame_1[1] - tempFrame_2[1]
                break
        while abs(deltaT) > 0.05:
            if deltaT > 0 and self.lock_2.acquire() and len(self.q_2) > 0:
                tempFrame_2 = self.q_2.pop()
                deltaT = tempFrame_1[1] - tempFrame_2[1]
            elif deltaT < 0 and self.lock_1.acquire() and len(self.q_1) > 0:
                tempFrame_1 = self.q_1.pop()
                deltaT = tempFrame_1[1] - tempFrame_2[1]
        print(tempFrame_1[1], tempFrame_2[1])
        return [tempFrame_1[0], tempFrame_2[0]]

if __name__ == "__main__":
    lock_1 = threading.Lock()
    lock_2 = threading.Lock()
    camera_1 = readCamera(0, lock_1)
    camera_2 = readCamera(1, lock_2)
    match = matchCamera(camera_1.q, camera_2.q, lock_1, lock_2)
    camera_1.start()
    camera_2.start()
    match.start()
    camera_1.join()
    camera_2.join()
    match.join()
    print("Done")