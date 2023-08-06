import os
import cv2 as cv
import argparse
import numpy as np
import json

class stereoCameraCalibration(object):
    def __init__(self, width, height, lattice):
        self.width = width
        self.height = height
        self.lattice = lattice

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteriaStereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    def stereoCalibration(self, imgL, imgR):
        objPoints = np.zeros((self.width * self.height, 3), np.float32)
        objPoints[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        # objPoints *= self.lattice

        imgPoints = []
        imgPointsL = []
        imgPointsR = []

        for i in range(len(imgL)):
            chessImgL = cv.imread(imgL[i], 0)
            chessImgR = cv.imread(imgR[i], 0)

            retL, cornerL = cv.findChessboardCorners(chessImgL, (self.width, self.height), cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FILTER_QUADS)
            retR, cornerR = cv.findChessboardCorners(chessImgR, (self.width, self.height), cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FILTER_QUADS)

            if retL and retR:
                imgPoints.append(objPoints)
                cv.cornerSubPix(chessImgL, cornerL, (11, 11), (-1, -1), self.criteria)
                cv.cornerSubPix(chessImgR, cornerR, (11, 11), (-1, -1), self.criteria)
                imgPointsL.append(cornerL)
                imgPointsR.append(cornerR)
                retCornerL = cv.drawChessboardCorners(chessImgL, (self.width, self.height), cornerL, retL)
                cv.imshow(imgL[i], chessImgL)
                cv.waitKey(2)
                retCornerR = cv.drawChessboardCorners(chessImgR, (self.width, self.height), cornerR, retR)
                cv.imshow(imgR[i], chessImgR)
                cv.waitKey(2)
        retL, matInterL, coeDisL, vecRL, vecTL = cv.calibrateCamera(imgPoints, imgPointsL, chessImgL.shape[::-1], None, None)
        retR, matInterR, coeDisR, vecRR, vecTR = cv.calibrateCamera(imgPoints, imgPointsR, chessImgR.shape[::-1], None, None)

        flags = 0
        flags |= cv.CALIB_FIX_INTRINSIC

        retS, matInterL, coeDisL, matInterR, coeDisR, R, T, E, F = cv.stereoCalibrate(imgPoints, imgPointsL, imgPointsR, matInterL, coeDisL, matInterR, coeDisR, chessImgL.shape[::-1], self.criteriaStereo, flags)
        cv.destroyAllWindows()
        totalErrorL = 0
        totalErrorR = 0
        for i in range(len(imgPoints)):
            imgPointsRepo, _ = cv.projectPoints(imgPoints[i], vecRL[i], vecTL[i], matInterL, coeDisL)
            error = cv.norm(imgPointsL[i], imgPointsRepo, cv.NORM_L2) / len(imgPointsRepo)
            totalErrorL += error
            imgPointsRepo, _ = cv.projectPoints(imgPoints[i], vecRR[i], vecTR[i], matInterR, coeDisR)
            error = cv.norm(imgPointsR[i], imgPointsRepo, cv.NORM_L2) / len(imgPointsRepo)
            totalErrorR += error
        print('Average error of repro L:', totalErrorL / len(imgPoints))
        print('Average error of repro R:', totalErrorR / len(imgPoints))
        return matInterL, coeDisL, matInterR, coeDisR, R, T

    def getRectifiedTransform(self, width, height, matInterL, coeDisL, matInterR, coeDisR, R, T):
        RL, RR, PL, PR, Q, roiL, roiR = cv.stereoRectify(matInterL, coeDisL, matInterR, coeDisR, (width, height), R, T, flags = cv.CALIB_ZERO_DISPARITY, alpha=0)
        mapLx, mapLy = cv.initUndistortRectifyMap(matInterL, coeDisL, RL, PL, (width, height), cv.CV_32FC1)
        mapRx, mapRy = cv.initUndistortRectifyMap(matInterR, coeDisR, RR, PR, (width, height), cv.CV_32FC1)
        return mapLx, mapLy, mapRx, mapRy, Q

    def getRectifiedImg(self, imgL, imgR, mapLx, mapLy, mapRx, mapRy):
        recImgL = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        recImgR = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        return recImgL, recImgR

    def drawLine(self, recImgL, recImgR):
        width = max(recImgL.shape[1], recImgR.shape[1])
        height = max(recImgL.shape[0], recImgR.shape[0])

        output = np.zeros((height, width*2, 3), np.uint8)
        output[0:recImgL.shape[0], 0:recImgL.shape[1]] = recImgL
        output[0:recImgR.shape[0], recImgL.shape[1]:] = recImgR

        lineInterval = 50
        for i in range(height // lineInterval):
            cv.line(output, (0, lineInterval * (i + 1)), (2 * width, lineInterval * (i + 1)), (0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        return output

def getParser():
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--width', type=int, default=11, help = 'CheckBoard Width size')
    parser.add_argument('--height', type=int, default=8, help='CheckBoard Height size')
    parser.add_argument('--lattice', type=float, default=0.02, help='Lattice Length')
    parser.add_argument('--imgDir', type=str, default='./img', help = 'Image Path')
    parser.add_argument('--saveDir', type=str, default='./config', help='Path to Save File')
    parser.add_argument('--fileName', type=str, default='stereoConfig', help='Stereo Config File Name')
    return parser

def getFile(path):
    imgPath = []
    for root, dirs, files in  os.walk(path):
        for file in files:
            imgPath.append(os.path.join(root, file))
    return imgPath

if __name__ == '__main__':
    args = getParser().parse_args()
    paramDict = {}
    fileL = getFile(args.imgDir + '/left')
    fileR = getFile(args.imgDir + '/right')
    imgL = cv.imread(fileL[2])
    imgR = cv.imread(fileR[2])
    height, width = imgL.shape[0:2]
    calibration = stereoCameraCalibration(args.width, args.height, args.lattice)
    matInterL, coeDisL, matInterR, coeDisR, R, T = calibration.stereoCalibration(fileL, fileR)
    mapLx, mapLy, mapRx, mapRy, Q = calibration.getRectifiedTransform(width, height, matInterL, coeDisL, matInterR, coeDisR, R, T)

    img_ = calibration.drawLine(imgL, imgR)
    cv.imshow('img', img_)
    recImgL, recImgR = calibration.getRectifiedImg(imgL, imgR, mapLx, mapLy, mapRx, mapRy)
    imgShow = calibration.drawLine(recImgL, recImgR)
    cv.imshow('output', imgShow)
    cv.waitKey(1)


    # paramDict['size'] = [width, height]
    paramDict['K1'] = matInterL.tolist()
    paramDict['D1'] = coeDisL.tolist()
    paramDict['K2'] = matInterR.tolist()
    paramDict['D2'] = coeDisR.tolist()
    # paramDict['mapLx'] = mapLx.tolist()
    # paramDict['mapLy'] = mapLy.tolist()
    # paramDict['mapRx'] = mapRx.tolist()
    # paramDict['mapRy'] = mapRy.tolist()
    paramDict['R'] = R.tolist()
    paramDict['T'] = T.tolist()
    paramDict['Q'] = Q.tolist()

    filePath = args.saveDir + '/' + args.fileName + '.json'
    with open(filePath, 'w') as f:
        json.dump(paramDict, f, indent=1)
    print('Done!')
    cv.destroyAllWindows()