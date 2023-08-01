import cv2 as cv
# import keyboard

def capture_image():
    # 打开摄像头
    cap = cv.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Can't open camera!")
        return
    img_count = 0
    # 从摄像头中读取一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read camera!")
            break
        cv.imshow('Camera', frame)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            cv.imwrite('./img/img_%d.png' % img_count, frame)
            img_count += 1

    # 关闭摄像头
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
