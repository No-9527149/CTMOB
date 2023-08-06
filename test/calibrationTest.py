import os
import cv2 as cv
import numpy as np


CORNER_SHAPE = (11, 8)
SIZE_PER_GRID = 0.02

def calibrationSingle(imgDir, configFile):
    width, height = CORNER_SHAPE
    cornerPointInt = np.zeros((width * height, 3), np.float32)
    cornerPointInt[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    cornerWorld = cornerPointInt * SIZE_PER_GRID

    objPoints = []
    imgPoints = []
    images = os.listdir(imgDir)
    for imgName in images:
        img = cv.imread(imgDir + '/' + imgName)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, cornerPointImage = cv.findChessboardCorners(imgGray, (width, height), None)
        if ret:
            objPoints.append(cornerWorld)
            imgPoints.append(cornerPointImage)
            cv.drawChessboardCorners(img, (width, height), cornerPointImage, ret)
            cv.imshow('Corners', img)
            cv.waitKey(1)
    cv.destroyAllWindows()

    ret, matInter, coeDis, vRot, vTrans = cv.calibrateCamera(objPoints, imgPoints, imgGray.shape[::-1], None, None)
    with open(configFile, 'w') as f:
        print(ret, file=f)
        print(matInter, file=f)
        print(coeDis, file=f)
        print(vRot, file=f)
        print(vTrans, file=f)
    total_error = 0
    for i in range(len(objPoints)):
        imgPointsRepo, _ = cv.projectPoints(objPoints[i], vRot[i], vTrans[i], matInter, coeDis)
        error = cv.norm(imgPoints[i], imgPointsRepo, cv.NORM_L2) / len(imgPointsRepo)
        total_error += error
    print('Average error of repro:', total_error / len(objPoints))
    return ret, matInter, coeDis, vRot, vTrans

def deDistortion(imgDir, saveDir, matInter, coeDis):
    images = os.listdir(imgDir)
    for imgName in images:
        img = cv.imread(imgDir + '/' + imgName)
        width, height = img.shape[:2]
        imgNew, roi = cv.getOptimalNewCameraMatrix(matInter, coeDis, (width, height), 0, (width, height))
        dst = cv.undistort(img, matInter, coeDis, None, imgNew)
        x, y, width, height = roi
        dst = dst[y:y + height, x:x + width]
        cv.imwrite(saveDir + os.sep + imgName, dst)
    print('Successfully Save')

if __name__ == '__main__':
    # left
    imgDir = "./img/left"
    ret_1, matInter_1, coeDis_1, vRot_1, vTrans_1 = calibrationSingle(imgDir, "./config/left.txt")
    saveDir = "./imgSave/left"
    if (not os.path.exists(saveDir)):
        os.makedirs(saveDir)
    deDistortion(imgDir, saveDir, matInter_1, coeDis_1)
    # right
    imgDir = "./img/right"
    ret_2, matInter_2, coeDis_2, vRot_2, vTrans_2 = calibrationSingle(imgDir, "./config/right.txt")
    saveDir = "./imgSave/right"
    if (not os.path.exists(saveDir)):
        os.makedirs(saveDir)
    deDistortion(imgDir, saveDir, matInter_2, coeDis_2)
    # stereo
    ret, matInter_1, coeDis_1, matInter_2, coeDis_2 = cv.stereoCalibrate()