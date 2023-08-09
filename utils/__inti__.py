import cv2 as cv
import numpy as np

def rotateAntiClockWise90ByNumpy(img):
    return np.rot90(img, -1)

def getRawImg(src_1, src_2):
    cap_1 = cv.VideoCapture(src_1)
    cap_2 = cv.VideoCapture(src_2)
    if not cap_1.isOpened() and not cap_2.isOpened():
        raise AssertionError("Can't open camera.")
    ret_1, img_1 = cv.imread(cap_1)
    ret_2, img_2 = cv.imread(cap_2)
    if not ret_1 and not ret_2:
        raise AssertionError("Can't read capture.") 
    return img_1, img_2