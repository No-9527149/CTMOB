import cv2 as cv
import time

def capture_image():
    # 打开摄像头
    cap = cv.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Can't open camera!")
        return

    timeStamps = []
    # 从摄像头中读取一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read camera!")
            break
        cv.imshow('Camera', frame)
        sysTime = time.time()
        timeStamps.append(cap.get(cv.CAP_PROP_POS_MSEC))
        if cv.waitKey(1) == 27:
            break

    # 关闭摄像头
    cap.release()
    cv.destroyAllWindows()
    print(timeStamps)

if __name__ == "__main__":
    capture_image()
