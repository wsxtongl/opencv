import cv2
import numpy as np
import matplotlib.pyplot as plt
#卷积滤波
# src = cv2.imread(r"./image/1.jpg")
# gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32) # 定义一个核
# dst = cv2.filter2D(gray, -1, kernel=kernel)
# cv2.imshow("src show", src)
# cv2.imshow("dst show", dst)
# cv2.waitKey(0)
#均值滤波,模糊
# src = cv2.imread(r"./image/1.jpg")
# dst = cv2.blur(src,(3,3))
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
#高斯滤波
# src = cv2.imread(r"./image/1.jpg")
# # 0：方差,方差越大越模糊
# dst = cv2.GaussianBlur(src,(5,5),1)
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
#中值滤波
# src = cv2.imread(r"./image/5.jpg")
# dst = cv2.medianBlur(src,5)
# cv2.imshow("src",src)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
#双边滤波
#去掉噪声并保留边缘纹理
#d:滤波邻域直径    sigmaColor：滤波的色彩空间参数    sigmaSpace ：滤波的距离空间参数
# src = cv2.imread(r"./image/1.jpg")
# dst1 = cv2.bilateralFilter(src,d=9,sigmaColor=75,sigmaSpace=75)
# cv2.imshow("src",src)
# cv2.imshow("dst1",dst1)
# cv2.waitKey(0)

#锐化,提高对比度
# src = cv2.imread(r"./image/1.jpg")
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
# dst = cv2.filter2D(src, -1, kernel=kernel)
# cv2.imshow("src show", src)
# cv2.imshow("dst show", dst)
# cv2.waitKey(0)
#USM锐化
# src = cv2.imread(r"./image/1.jpg")
# dst = cv2.GaussianBlur(src, (5, 5), 2)
# dst = cv2.addWeighted(src, 2, dst, -1, 0)
# cv2.imshow("src", src)
# cv2.imshow("ds show", dst)
# cv2.waitKey(0)
#梯度算子
#提取轮廓
#Sobel:求一阶，高斯平滑与微分操作的结合体，所以它的抗噪声能力很好
#Scharr：求二阶，对Sobel滤波器的改进版本，梯度变得更陡
#Laplacian:x,y两者兼顾

# src= cv2.imread(r"./image/1.jpg",0)
# sobel_x = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=5)
# sobel_y = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=5)
# scharr_x = cv2.Scharr(src,cv2.CV_64F,1,0)
# scharr_y = cv2.Scharr(src,cv2.CV_64F,0,1)
# laplacian = cv2.Laplacian(src,cv2.CV_64F)
# plt.subplot(2,3,1)
# plt.imshow(src,cmap='gray')
# plt.title("src")
# plt.axis('off')
# plt.subplot(2,3,2)
# plt.imshow(sobel_x,cmap='gray')
# plt.title("sobel_x")
# plt.axis('off')
# plt.subplot(2,3,3)
# plt.imshow(sobel_y,cmap='gray')
# plt.title("sobel_y")
# plt.axis('off')
# plt.subplot(2,3,4)
# plt.imshow(laplacian,cmap='gray')
# plt.title("laplacian")
# plt.xticks([]),plt.yticks([])
# plt.subplot(2,3,5)
# plt.imshow(scharr_x,cmap='gray')
# plt.title("scharr_x")
# plt.axis('off')
# plt.subplot(2,3,6)
# plt.imshow(scharr_y,cmap='gray')
# plt.title("scharr_y")
# plt.axis('off')
# plt.show()
#

# img = cv2.imread(r"./image/1.jpg")
#
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # img_hsv[:,:,2] =0.8 * img_hsv[:,:,2]
# img_hsv[0,:,:] =0.2 * img_hsv[0,:,:]
#
# img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
# dst = cv2.GaussianBlur(img_bgr, (3, 3), 1)
# dst = cv2.addWeighted(img_bgr, 2, dst, -1, 0)

# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
# dst = cv2.filter2D(src, -1, kernel=kernel)

# ds = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,element,iterations=1)
# laplacian = cv2.Laplacian(dst,-1)
# canny = cv2.Canny(dst,50,150)
# cv2.imshow("img",img)
# cv2.imshow("img_bgr",img_bgr)
# cv2.imshow("dst show", dst)
# cv2.imshow("lap", laplacian)
# cv2.imshow("canny",canny)
#
# cv2.waitKey(0)
#提取边缘
# img = cv2.imread("./image/23.jpg")
# img1 = cv2.convertScaleAbs(img,alpha =10,beta=0)
# gauss = cv2.GaussianBlur(img1,(5,5),2)
#
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# dst = cv2.morphologyEx(gauss,cv2.MORPH_CLOSE,element,iterations=1)
# dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,element,iterations=1)
# canny = cv2.Canny(dst,100,150)
# cv2.imshow("canny",canny)
# cv2.imshow("img",img)
# cv2.waitKey(0)

# img = cv2.imread("./image/24.jpg")
#
# # x = cv2.Scharr(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
# # absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Sobel = cv2.addWeighted(absY, 0.5, absY, 0.5, 0)
# gauss = cv2.GaussianBlur(absY,(5,5),1)
# dst1 = cv2.addWeighted(img, 1.2, gauss, -1.2, 0)
#
#
#
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# dst = cv2.morphologyEx(dst1,cv2.MORPH_CLOSE,element,iterations=1)
# dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,element,iterations=1)
# canny = cv2.Canny(dst,100,150)
# cv2.imshow("canny",canny)
# cv2.imshow("img",img)
#
# cv2.waitKey(0)
