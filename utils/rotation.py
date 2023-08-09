import numpy as np
import cv2 as cv
import os

def rotateAntiClockWise90ByNumpy(img_file):
    img = cv.imread(img_file)
    img90 = np.rot90(img, -1)
    cv.imshow("rotate", img90)
    cv.waitKey(1)
    return img90

if __name__ == '__main__':
    leftDir = './test/img_1_test.png'
    rightDir = './test/img_2_test.png'
    cv.imwrite(leftDir, rotateAntiClockWise90ByNumpy(leftDir))
    cv.imwrite(rightDir, rotateAntiClockWise90ByNumpy(rightDir)) 
    # for root, dirs, files in os.walk(leftDir):
    #     for file in files:
    #         if file == '.DS_Store':
    #             continue
    #         cv.imwrite(leftDir + os.sep + file, rotateAntiClockWise90ByNumpy(leftDir + os.sep + file))
    # for root, dirs, files in os.walk(rightDir):
    #     for file in files:
    #         if file == '.DS_Store':
    #             continue
    #         cv.imwrite(rightDir + os.sep + file, rotateAntiClockWise90ByNumpy(rightDir + os.sep + file))
    cv.destroyAllWindows()