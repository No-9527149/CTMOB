import time
import cv2 as cv
from threading import Thread
from threading import Event
from collections import deque


class synchronization(object):
    def __init__(self, src_1, src_2, event):
        self.src_1 = src_1
        self.src_2 = src_2
        self.event = event
        self.q_1 = deque(maxlen=30)
        self.q_2 = deque(maxlen=30)

    def readCamera(self):
        print("Begin camera reading...")
        cap_1 = cv.VideoCapture(self.src_1)
        cap_2 = cv.VideoCapture(self.src_2)
        while cap_1.isOpened() and cap_2.isOpened():
            sysTime = time.time()
            _, frame_1 = cap_1.read()
            self.q_1.append([frame_1, sysTime])
            sysTime = time.time()
            _, frame_2 = cap_2.read()
            self.q_2.append([frame_2, sysTime])
            print("------Reading-------")
        #     # cv.imshow will block the thread
        #     cv.imshow('1', frame_1)
        #     cv.imshow('2', frame_2)
        #     if cv.waitKey(1) == 27:
        #         break
        # cv.destroyAllWindows()
        # time.sleep(2)
        print('Camera reading done')
        cap_1.release()
        cap_2.release()
        self.event.set()

    def matchCamera(self):
        print('Begin camera matching...')
        while True:
            deltaT = 0
            while True:
                if len(self.q_1) > 0 and len(self.q_2) > 0:
                    tempFrame_1 = self.q_1.pop()
                    tempFrame_2 = self.q_2.pop()
                    deltaT = tempFrame_1[1] - tempFrame_2[1]
                    break
            while abs(deltaT) > 0.03:
                if deltaT > 0 and len(self.q_2) > 0:
                    tempFrame_2 = self.q_2.pop()
                    deltaT = tempFrame_1[1] - tempFrame_2[1]
                elif deltaT < 0 and len(self.q_1) > 0:
                    tempFrame_1 = self.q_1.pop()
                    deltaT = tempFrame_1[1] - tempFrame_2[1]
            print("Camera_1 timestamp: %f" % tempFrame_1[1])
            print("Camera_2 timestamp: %f" % tempFrame_2[1])
            if self.event.is_set():
                break
        print("Frame matching done")

    def run(self):
        camera = Thread(target=self.readCamera)
        match = Thread(target=self.matchCamera)
        match.start()
        camera.start()
        match.join()
        camera.join()


if __name__ == '__main__':
    event = Event()
    event.clear()
    combine = synchronization(0, 1, event)
    combine.run()
