import cv2 as cv
import numpy as np

def rotateAntiClockWise90ByNumpy(img):
    img90 = np.rot90(img, -1)
    return img90

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
        frame_1 = rotateAntiClockWise90ByNumpy(frame_1)
        frame_2 = rotateAntiClockWise90ByNumpy(frame_2)
        cv.imshow('Camera_1', frame_1)
        cv.imshow('Camera_2', frame_2)
        key = cv.waitKey(1)
        if key == 27 or img_count == 1:
            break
        elif key == ord('s'):
            # test picture
            cv.imwrite('./test/img_1_test.png', frame_1)
            cv.imwrite('./test/img_2_test.png', frame_2)
            # # calibration picture
            # cv.imwrite('./img/left/img_1_%d.png' % img_count, frame_1)
            # cv.imwrite('./img/right/img_2_%d.png' % img_count, frame_2)
            img_count += 1

    # Release Camera
    cap_1.release()
    cap_2.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
