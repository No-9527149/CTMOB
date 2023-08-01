import os
import cv2 as cv
import numpy as np
import glob


CORNER_SHAPE = (11, 8)
SIZE_PER_GRID = 0.02

def calibrationSingle(imgDir):
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
    print(('ret:'), ret)
    print(('internal matrix:\n'), matInter)
    print(('distortion coefficients:\n'), coeDis)
    print(('rotation vectors:\n'), vRot)
    print(('transformation vectors:\n'), vTrans)

    total_error = 0
    for i in range(len(objPoints)):
        imgPointsRepo, _ = cv.projectPoints(objPoints[i], vRot[i], vTrans[i], matInter, coeDis)
        error = cv.norm(imgPoints[i], imgPointsRepo, cv.NORM_L2) / len(imgPointsRepo)
        total_error += error
    print('Average error of repro:', total_error / len(objPoints))

    return matInter, coeDis

def deDistortion(imgDir, saveDir, matInter, coeDis):
    images = os.listdir(imgDir)
    for imgName in images:
        img = cv.imread(imgDir + '/' + imgName)
        width, height = img.shape[:2]
        imgNew, roi = cv.getOptimalNewCameraMatrix(matInter, coeDis, (width, height), 0, (width, height))
        dst = cv.undistort(img, matInter, coeDis, None, imgNew)
        # x, y, width, height = roi
        # dst = dst[y:y + height, x:x + width]
        cv.imwrite(saveDir + os.sep + imgName, dst)
    print('Successfully Save')

if __name__ == '__main__':
    imgDir = "./img"
    matInter, coeDis = calibrationSingle(imgDir)
    saveDir = "./imgSave"
    if (not os.path.exists(saveDir)):
        os.makedirs(saveDir)
    deDistortion(imgDir, saveDir, matInter, coeDis)