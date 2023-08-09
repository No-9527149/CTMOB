import time
import open3d as o3d
import numpy as np
import cv2 as cv
# from collections import deque
from multiprocessing import Process
from multiprocessing import Event
from multiprocessing import Queue
from config.stereoConfigPython import stereoConfig

SRC_LEFT = 0
SRC_RIGHT = 1

MAXLENGTH = 5


def getRawImg(event, rawImgQL, rawImgQR, synImgQL, synImgQR, src_1=0, src_2=1):
    print("Begin Camera Reading...")
    cap_1 = cv.VideoCapture(src_1)
    cap_2 = cv.VideoCapture(src_2)
    _, img_1 = cap_1.read()
    synImgQL.put(img_1)
    _, img_2 = cap_2.read()
    synImgQR.put(img_2)
    while cap_1.isOpened() and cap_2.isOpened():
        if event.is_set():
            continue
        print("------Reading-------")
        sysTime = time.time()
        _, img_1 = cap_1.read()
        rawImgQL.put([img_1, sysTime])
        sysTime = time.time()
        _, img_2 = cap_2.read()
        rawImgQR.put([img_2, sysTime])
    cap_1.release()
    cap_2.release()


def synchronizeImg(event, rawImgQL, rawImgQR, synImgQL, synImgQR):
    print('Begin Image Synchronize...')
    while True:
        if event.is_set():
            continue
        deltaT = 0
        tempFrame_1 = rawImgQL.get()
        tempFrame_2 = rawImgQR.get()
        deltaT = tempFrame_1[1] - tempFrame_2[1]
        while abs(deltaT) > 0.03:
            if deltaT > 0:
                tempFrame_2 = rawImgQR.get()
                deltaT = tempFrame_1[1] - tempFrame_2[1]
            elif deltaT < 0:
                tempFrame_1 = rawImgQL.get()
                deltaT = tempFrame_1[1] - tempFrame_2[1]
        print("Camera_1 timestamp: %f" % tempFrame_1[1])
        print("Camera_2 timestamp: %f" % tempFrame_2[1])
        synImgQL.put(tempFrame_1[0])
        synImgQR.put(tempFrame_2[0])


class cloudPoint(stereoConfig):
    def __init__(self):
        super(cloudPoint).__init__()
        stereoConfig.__init__(self)

    def run(self, event, synImgQL, synImgQR):
        event.set()
        tracker = cv.TrackerKCF_create()
        imgL = synImgQL.get()
        imgR = synImgQR.get()
        # Define an initial bounding box
        bbox = (287, 23, 86, 320)
        # Uncomment the line below to select a different bounding box
        bbox = cv.selectROI(imgL, False)
        tracker.init(imgL, bbox)
        event.clear()

        print('Begin Calculating PCD...')
        while True:
            imgL = synImgQL.get()
            imgR = synImgQR.get()
            height, width = imgL.shape[0:2]

            _, bbox = tracker.update(imgL)
            # Draw bounding box
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            cv.rectangle(imgL, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xx = round((2 * x + w) / 2)
            yy = round((2 * y + h) / 2)

            disp, _ = self.stereoMatchSGBM(imgL, imgR, False)
            dot_disp = disp[yy][xx]
            xr = xx + dot_disp
            yr = yy

            # Calculate depth
            # _, _, _, _, Q = self.getRectifyTransform(height, width)
            # z = self.getDepthMapWithQ(disp, Q)[yy][xx]
            z = self.getDepthMapWithConfig(dot_disp)

            text = str(xr) + ',' + str(yr) + ',' + str(z)
            cv.putText(imgL, text, (x, y), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            cv.imshow('imgL', imgL)
            cv.imshow('imgR', imgR)
            key = cv.waitKey(1)
            if key == 27:
                event.set()
                break
            elif key == ord('s'):
                event.set()
                bbox = (287, 23, 86, 320)
                bbox = cv.selectROI(imgL, False)
                tracker.init(imgL, bbox)
                event.clear()
        '''
        # Open3D draw cloudPoint
        colorImage = o3d.geometry.Image(imgL)
        depthImage = o3d.geometry.Image(depthMap)
        rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(
            colorImage, depthImage, depth_scale=1000.0, depth_trunc=np.inf)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()

        # getDepthMapWith Q
        fx = Q[2, 3]
        fy = Q[2, 3]
        cx = Q[0, 3]
        cy = Q[1, 3]

        intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsics = np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
        pointCloud = o3d.geometry.PointCloud().create_from_rgbd_image(
            rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
        o3d.visualization.draw_geometries([pointCloud], width=720, height=480)
        o3d.io.write_point_cloud("PointCloud_%s.pcd" % time.asctime(time.localtime(time.time())), pointCloud)
        '''

    def preProcess(self, img_1, img_2):
        '''
        Preprocess to get rid of the effects of lighting
        '''
        # Color Image -> Gray Image
        if (img_1.ndim == 3):
            img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
        if (img_2.ndim == 3):
            img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

        # Histogram Equalization
        img_1 = cv.equalizeHist(img_1)
        img_2 = cv.equalizeHist(img_2)
        return img_1, img_2

    def unDistortion(self, image, cameraMatrix, distCoef):
        '''
        unDistortion
        '''
        unDistortionImg = cv.undistort(image, cameraMatrix, distCoef)
        return unDistortionImg

    def getRectifyTransform(self, height, width):
        '''
        Get the mapping transformation matrix and reprojection matrix 
        for distortion correction and stereo correction
        '''
        # Read Internal Params & External Params
        leftK = self.cam_matrix_left
        rightK = self.cam_matrix_right
        leftDistortion = self.distortion_l
        rightDistortion = self.distortion_r
        R = self.R
        T = self.T

        # Calculate Rectified Image
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(leftK, leftDistortion, rightK, rightDistortion,
                                                        (width, height), R, T, alpha=0)
        map_1x, map_1y = cv.initUndistortRectifyMap(leftK, leftDistortion, R1, P1, (width, height), cv.CV_32FC1)
        map_2x, map_2y = cv.initUndistortRectifyMap(rightK, rightDistortion, R2, P2, (width, height), cv.CV_32FC1)
        return map_1x, map_1y, map_2x, map_2y, Q

    def rectifyImage(self, img_1, img_2, map_1x, map_1y, map_2x, map_2y): 
        '''
        Distortion correction & stereo correction
        '''
        rectifiedImg1 = cv.remap(img_1, map_1x, map_1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        rectifiedImg2 = cv.remap(img_2, map_2x, map_2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        return rectifiedImg1, rectifiedImg2

    def drawLine(self, img_1, img_2):
        '''
        Check results of stereo correction
        '''
        height = max(img_1.shape[0], img_2.shape[0])
        width = img_1.shape[1] + img_2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
        output[0:img_2.shape[0], img_1.shape[1]:] = img_2

        line_interval = 50
        for k in range(height // line_interval):
            cv.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        return output

    def stereoMatchSGBM(self, imgL, imgR, downScale=False):
        '''
        Calculate parallax
        '''
        # Set params
        if imgL.ndim == 2:
            imgChannels = 1
        else:
            imgChannels = 3

        blockSize = 3
        paramL = {'minDisparity': 0,
                    'numDisparities': 128,
                    'blockSize': blockSize,
                    'P1': 8 * imgChannels * blockSize ** 2,
                    'P2': 32 * imgChannels * blockSize ** 2,
                    'disp12MaxDiff': 1,
                    'preFilterCap': 63,
                    'uniquenessRatio': 15,
                    'speckleWindowSize': 100,
                    'speckleRange': 1,
                    'mode': cv.STEREO_SGBM_MODE_SGBM_3WAY
                    }

        # Set SGBM object
        leftMatcher = cv.StereoSGBM_create(**paramL)
        paramR = paramL
        paramR['minDisparity'] = -paramL['numDisparities']
        rightMatcher = cv.StereoSGBM_create(**paramR)

        # Calculate parallax
        size = (imgL.shape[1], imgL.shape[0])
        if downScale == False:
            disparityL = leftMatcher.compute(imgL, imgR)
            disparityR = rightMatcher.compute(imgR, imgL)
        else:
            leftImgDown = cv.pyrDown(imgL)
            rightImgDown = cv.pyrDown(imgR)
            factor = imgL.shape[1] / leftImgDown.shape[1]

            disparityLeftHalf = leftMatcher.compute(leftImgDown, rightImgDown)
            disparityRightHalf = rightMatcher.compute(rightImgDown, leftImgDown)
            disparityL = cv.resize(disparityLeftHalf, size, interpolation=cv.INTER_AREA)
            disparityR = cv.resize(disparityRightHalf, size, interpolation=cv.INTER_AREA)
            disparityL = factor * disparityL
            disparityR = factor * disparityR

        # Real Parallax
        trueDispL = disparityL.astype(np.float32) / 16.
        trueDispR = disparityR.astype(np.float32) / 16.

        return trueDispL, trueDispR

    def hw3ToN3(self, points):
        '''
        Convert h * w * 3 array to N * 3 array
        '''
        height, width = points.shape[0:2]

        points_1 = points[:, :, 0].reshape(height * width, 1)
        points_2 = points[:, :, 1].reshape(height * width, 1)
        points_3 = points[:, :, 2].reshape(height * width, 1)

        points_ = np.hstack((points_1, points_2, points_3))

        return points_

    def depthColor2Cloud(self, points3d, colors):
        '''
        Get cloudPoints
        '''
        rows, cols = points3d.shape[0:2]
        size = rows * cols

        points_ = self.hw3ToN3(points3d)
        colors_ = self.hw3ToN3(colors).astype(np.int64)

        # Information about color
        blue = colors_[:, 0].reshape(size, 1)
        green = colors_[:, 1].reshape(size, 1)
        red = colors_[:, 2].reshape(size, 1)

        rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

        # Overlay coordinates & colors as cloudPoint
        pointCloud = np.hstack((points_, rgb)).astype(np.float32)

        # Del inappropriate points
        X = pointCloud[:, 0]
        Y = pointCloud[:, 1]
        Z = pointCloud[:, 2]

        # Empirical Prams
        # Need to be adjusted according to actual situation
        removeIdx1 = np.where(Z <= 0)
        removeIdx2 = np.where(Z > 15000)
        removeIdx3 = np.where(X > 10000)
        removeIdx4 = np.where(X < -10000)
        removeIdx5 = np.where(Y > 10000)
        removeIdx6 = np.where(Y < -10000)
        removeIdx = np.hstack(
            (removeIdx1[0], removeIdx2[0], removeIdx3[0], removeIdx4[0], removeIdx5[0], removeIdx6[0]))

        pointCloud_1 = np.delete(pointCloud, removeIdx, 0)

        return pointCloud_1

    def getDepthMapWithQ(self, disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
        points_3d = cv.reprojectImageTo3D(disparityMap, Q)
        depthMap = points_3d[:, :, 2]
        reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
        depthMap[reset_index] = 0
        return depthMap.astype(np.float32)

    def getDepthMapWithConfig(self, dot_disp) -> np.ndarray:
        fb = self.cam_matrix_left[0, 0] * (-self.T[0])
        doffs = self.doffs
        # depthMap = np.divide(fb, disparityMap + doffs)
        # resetIndex_1 = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
        # depthMap[resetIndex_1] = 0
        # resetIndex_2 = np.where(disparityMap < 0.0)
        # depthMap[resetIndex_2] = 0
        # return depthMap.astype(np.float32)
        depth = fb / (dot_disp + doffs)
        return depth


if __name__ == '__main__':
    event = Event()
    event.clear()

    rawImgQL = Queue(MAXLENGTH)
    rawImgQR = Queue(MAXLENGTH)
    synImgQL = Queue(MAXLENGTH)
    synImgQR = Queue(MAXLENGTH)

    ptCloud = cloudPoint()
    camera = Process(target=getRawImg, args=(event, rawImgQL, rawImgQR, synImgQL, synImgQR, SRC_LEFT, SRC_RIGHT))
    synchronize = Process(target=synchronizeImg, args=(event, rawImgQL, rawImgQR, synImgQL, synImgQR))
    depth = Process(target=ptCloud.run, args=(event, synImgQL, synImgQR))

    camera.start()
    synchronize.start()
    depth.start()

    depth.join()
    camera.terminate()
    synchronize.terminate()

    print("Done!")