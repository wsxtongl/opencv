import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
rawImage = cv2.imread("./image/car.jpg")
# 高斯模糊，将图片平滑化，去掉干扰的噪声
image = cv2.GaussianBlur(rawImage, (3, 3), 0)
# 图片灰度化
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Sobel算子（X方向）
Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
# Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(Sobel_x) # 转回uint8

# absY = cv2.convertScaleAbs(Sobel_y)
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
image = absX
# 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
ret, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel = np.ones((3,3),np.int8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 膨胀腐蚀(形态学处理)
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)
# 平滑处理，中值滤波
image1 = cv2.medianBlur(image, 15)
cv2.imshow('image1', image1)
# 查找轮廓
contours, w1 = cv2.findContours(image1, cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    x,y,w,h = cv2.boundingRect(item)
    if w > (h * 2):
        # 裁剪区域图片
        chepai = rawImage[y:y + h, x:x + w]
        cv2.imshow('chepai', chepai)
# 绘制轮廓
image = cv2.drawContours(rawImage, contours, -1, (0, 0, 255), 1)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()