import cv2 as cv

def capture_image():
    cap_1 = cv.VideoCapture(0)
    cap_2 = cv.VideoCapture(1)

    if not cap_1.isOpened() or not cap_2.isOpened():
        print("Can't open camera!")
        return
    img_count = 0

    while True:
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        if not ret_1 or not ret_2:
            print("Can't read camera!")
            break
        cv.imshow('Camera_1', frame_1)
        cv.imshow('Camera_2', frame_2)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            cv.imwrite('./img/left/img_1_%d.png' % img_count, frame_1)
            cv.imwrite('./img/right/img_2_%d.png' % img_count, frame_2)
            img_count += 1

    # Release Camera
    cap_1.release()
    cap_2.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
