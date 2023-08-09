# -*- coding: utf-8 -*-
import sys
import cv2 as cv
import numpy as np
import config.stereoConfigPython as stereoConfig
# import config.stereoConfigMatlab as stereoConfig
import open3d as o3d


# 预处理
def preProcess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    if (img2.ndim == 3):
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    # 直方图均衡
    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)
    return img1, img2


# 消除畸变
def unDistortion(image, cameraMatrix, distCoef):
    unDistortionImg = cv.undistort(image, cameraMatrix, distCoef)
    return unDistortionImg


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    leftK = config.cam_matrix_left
    rightK = config.cam_matrix_right
    leftDistortion = config.distortion_l
    rightDistortion = config.distortion_r
    R = config.R
    T = config.T
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(leftK, leftDistortion, rightK, rightDistortion,
                                                    (width, height), R, T, alpha=0)
    map1x, map1y = cv.initUndistortRectifyMap(leftK, leftDistortion, R1, P1, (width, height), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(rightK, rightDistortion, R2, P2, (width, height), cv.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y): 
    rectifiedImg1 = cv.remap(image1, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    rectifiedImg2 = cv.remap(image2, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    return rectifiedImg1, rectifiedImg2


# 立体校正检验----画线
def drawLine(img_1, img_2):
    height = max(img_1.shape[0], img_2.shape[0])
    width = img_1.shape[1] + img_2.shape[1]
    
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    output[0:img_2.shape[0], img_1.shape[1]:] = img_2
    
    line_interval = 50
    for k in range(height // line_interval):
        cv.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    return output


# 视差计算
def stereoMatchSGBM(leftImage, rightImage, downScale=False):
    # SGBM匹配参数设置
    if leftImage.ndim == 2:
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

    # 构建SGBM对象
    leftMatcher = cv.StereoSGBM_create(**paramL)
    paramR = paramL
    paramR['minDisparity'] = -paramL['numDisparities']
    rightMatcher = cv.StereoSGBM_create(**paramR)

    # 计算视差图
    size = (leftImage.shape[1], leftImage.shape[0])
    if downScale == False:
        disparityL = leftMatcher.compute(leftImage, rightImage)
        disparityR = rightMatcher.compute(rightImage, leftImage)
    else:
        leftImgDown = cv.pyrDown(leftImage)
        rightImgDown = cv.pyrDown(rightImage)
        factor = leftImage.shape[1] / leftImgDown.shape[1]

        disparityLeftHalf = leftMatcher.compute(
            leftImgDown, rightImgDown)
        disparityRightHalf = rightMatcher.compute(
            rightImgDown, leftImgDown)
        disparityL = cv.resize(
            disparityLeftHalf, size, interpolation=cv.INTER_AREA)
        disparityR = cv.resize(
            disparityRightHalf, size, interpolation=cv.INTER_AREA)
        disparityL = factor * disparityL
        disparityR = factor * disparityR

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDispL = disparityL.astype(np.float32) / 16.
    trueDispR = disparityR.astype(np.float32) / 16.

    return trueDispL, trueDispR


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def depthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointCloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointCloud[:, 0]
    Y = pointCloud[:, 1]
    Z = pointCloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
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


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoConfig.stereoConfig) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    resetIndex_1 = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[resetIndex_1] = 0
    resetIndex_2 = np.where(disparityMap < 0.0)
    depthMap[resetIndex_2] = 0
    return depthMap.astype(np.float32)


if __name__ == '__main__':
    # Load img
    imgL = cv.imread('./test/img_1_test.png')
    imgR = cv.imread('./test/img_2_test.png')
    init = drawLine(imgL, imgR)
    cv.imshow('init', init)
    if (imgL is None) or (imgR is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    height, width = imgL.shape[0:2]

    # Load config
    config = stereoConfig.stereoConfig()
    print(config.cam_matrix_left)

    # Stereo Correction
    # # Get mapping matrix for distortion correction and stereo correction
    # # Get reprojection matrix for computing pixel space coordinates
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    rectifiedImgL, rectifiedImgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
    print(Q)

    # Draw Lines
    line = drawLine(rectifiedImgL, rectifiedImgR)
    cv.imshow('lines', line)

    # Stereo Match
    # # Preprocess: Weaken the influence of uneven illumination
    imgL_, imgR_ = preProcess(imgL, imgR)
    disp, _ = stereoMatchSGBM(imgL, imgR, False)
    cv.imshow('match', disp * 4)

    # Calculate depth map
    depthMap = getDepthMapWithQ(disp, Q)
    # depthMap = getDepthMapWithConfig(disp, config)

    minDepth = np.min(depthMap)
    maxDepth = np.max(depthMap)
    print(minDepth, maxDepth)
    depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)
    cv.imshow("DepthMap", depthMapVis)
    cv.waitKey(0)

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
    
    # getDepthMapWith config
    # fx = config.cam_matrix_left[0, 0]
    # fy = fx
    # cx = config.cam_matrix_left[0, 2]
    # cy = config.cam_matrix_left[1, 2]
    print(fx, fy, cx, cy)
    intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsics = np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
    pointCloud = o3d.geometry.PointCloud().create_from_rgbd_image(
        rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
    o3d.io.write_point_cloud("PointCloud.pcd", pointCloud)
    o3d.visualization.draw_geometries([pointCloud], width=720, height=480)
    cv.destroyAllWindows()
    sys.exit(0)
'''
def view_cloud(pointCloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointCloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass
'''
'''
    # pcl

    # Calculate 3D coordinates of the pixels (left camera coordinate system)
    # Q: Reprojection matrix in function getRectifyTransform()
    points_3d = cv.reprojectImageTo3D(disp, Q)

    # CloudPoint: Point_XYZRGBA format
    pointCloud = DepthColor2Cloud(points_3d, imgL)

    view_cloud(points_3d)
'''
