import cv2
import numpy as np

CUBE = 90
ROW_CORNER = 9
COL_CORNER = 12

img = np.zeros((ROW_CORNER * CUBE, COL_CORNER * CUBE, 1))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i, j] = 255
        if (int(i/CUBE) % 2 == 0) and (int(j/CUBE) % 2 == 0):
            img[i, j] = 0
        if (int(i/CUBE) % 2 == 1) and (int(j/CUBE) % 2 == 1):
            img[i, j] = 0
cv2.imwrite("checkBoard.png", img)