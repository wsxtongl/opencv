import numpy as np
import cv2
import matplotlib.pyplot as plt
#去水印
#
# img = cv2.imread(r"./image/13.jpg")
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_blue = np.array([100, 43, 46])
# upper_blue = np.array([124, 255, 255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# scan = np.ones((3,3),np.uint8)
# cor = cv2.dilate(mask,scan,iterations=1)
# cor1 = cv2.dilate(mask,element,iterations=2)
# specular = cv2.inpaint(img,cor1,3,flags=cv2.INPAINT_TELEA)
# cv2.imshow("img",img)
# cv2.imshow("mask",mask)
# cv2.imshow("specular",specular)
# cv2.waitKey(0)






def edge_detect(img):
    # 高斯模糊,降低噪声
    img[:, :, 2] = 1.5 * img[:, :, 2]
    blurred = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 2)

    # 灰度图像
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    dst = cv2.GaussianBlur(gray, (5, 5), 3)
    dst = cv2.addWeighted(gray, 3, dst, -1, 0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # iterations:迭代次数
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element, iterations=1)

    # 图像梯度
    xgrad = cv2.Sobel(dst, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(dst, cv2.CV_16SC1, 0, 1)
    # 计算边缘
    # 50和150参数必须符合1：3或者1：2
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    dst = cv2.bitwise_and(img, img, mask=edge_output)

    return edge_output, dst


img = cv2.imread('./image/24.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

edge_output, canny_edge = edge_detect(img.copy())

cv2.imshow("edg",edge_output)
cv2.imshow("canny",canny_edge)
cv2.waitKey(0)




