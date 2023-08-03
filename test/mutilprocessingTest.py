import os
import time

import cv2
import gc
from multiprocessing import Process, Manager


class MyCapture():
    """
       :param cam: 摄像头参数
       :param stack: Manager.list对象
       :param top: 缓冲栈容量
       :return: None
    """

    def __init__(self, stack, cam, top):
        self.stack = stack
        self.cam = cam
        self.top = top

    # 向共享缓冲栈中写入数据:
    def write(self) -> None:
        print('Process to write: %s' % os.getpid())
        cap = cv2.VideoCapture(self.cam)
        cv2.namedWindow('ip_camera', flags=cv2.WINDOW_NORMAL |
                        cv2.WINDOW_FREERATIO)
        if not cap.isOpened():
            print('请检查IP地址还有端口号，或者查看IP摄像头是否开启')
        while cap.isOpened():
            ret, frame = cap.read()
            self.stack.append(frame)
            if len(self.stack) >= self.top:
                del self.stack[:]
                gc.collect()
            cv2.imshow('ip_camera %d' % self.cam, frame)
            if cv2.waitKey(1) == ord('q'):
                # 退出程序
                break
        print("写进程退出！！！！")
        cv2.destroyWindow('ip_camera %d' % self.cam)
        cap.release()

    # 在缓冲栈中读取数据:
    def read(self) -> None:
        print('Process to read: %s' % os.getpid())

        start_time = time.time()
        counter = 0

        while True:
            if len(self.stack) != 0:
                value = self.stack.pop()

                counter += 1  # 计算帧数
                cv2.putText(value, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                            3)

                gray_img = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)  # 灰度转换
                cv2.imshow("img %d" % self.cam, gray_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        print("读进程退出！！！！")
        cv2.destroyWindow('img %d' % self.cam)

    def multi_capture(self):
        pw = Process(target=self.write)
        pr = Process(target=self.read)
        # 启动子进程pw，写入:
        pw.start()
        # 启动子进程pr，读取:
        pr.start()
        # 等待pr结束:
        pr.join()
        # pw进程里是死循环，无法等待其结束，只能强行终止:
        print("写进程强行中止！！！！")
        pw.terminate()


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q_1 = Manager().list()
    # q_2 = Manager().list
    mc_1 = MyCapture(q_1, 1, 30)
    # mc_2 = MyCapture(q_2, 1, 30)
    mc_1.multi_capture()
    # mc_2.multi_capture()