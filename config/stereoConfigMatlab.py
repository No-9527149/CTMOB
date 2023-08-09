import numpy as np
# 双目相机参数


class stereoCamera(object):
    def __init__(self):
        # Left Camera Internal Matrix
        self.cam_matrix_left = np.array([[  1.010023055338315e+03,  1.843146351214746,     9.756503220367209e+02],
                                        [   0.,                     1.007307286987820e+03,  5.560523268176092e+02],
                                        [   0.,                     0.,                     1.]])

        # Right Camera Internal Matrix
        self.cam_matrix_right = np.array([[ 1.001640100274601e+03,  3.456556357710328,      9.494897235048558e+02],
                                        [   0.,                     9.995988167286608e+02,  5.420784184708150e+02],
                                        [   0.,                     0.,                     1.]])

        # Left/Right Distortion Coefficient [k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.175575985605628, -0.305977422058846, 0.001131595379581, -0.002381689156677, 0.132913868517007]])
        self.distortion_r = np.array([[0.173322387342150, -0.294906223306459, 0.001530959430033,  -0.001067799648692, 0.125723331514175]])

        # Rotation Matrix
        self.R = np.array([[0.999997903823885,      -4.013055827027291e-04,  0.002007810166663],
                        [   4.058552228470703e-04, 0.999997350084336,      -0.002266077192843],
                        [   -0.002006895456707,     0.002266887322989,      0.999995416785742]])

        # Transition Matrix
        self.T = np.array([[1.060243567303746], [-54.390072833660900], [-1.273502832790520]])

        # Baseline Distance(mm)
        self.baseline = 1.060243567303746

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False
