import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# img = cv2.imread('./image/25.jpg')
# imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,w1 = cv2.findContours(thresh, cv2.RETR_TREE,
# cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours[0]))
#重心
# M = cv2.moments(contours[0])
# print(M['m10']/M['m00'],M['m01']/M['m00'])
# epsilon = 90.5#精度
# approx = cv2.approxPolyDP(contours[0],epsilon,True)
# img_contour= cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
# img_contour= cv2.drawContours(img, contours, 0, (0, 255, 0), 1)
#判断凸包
# hull = cv2.convexHull(contours[0])
# print(cv2.isContourConvex(contours[0]), cv2.isContourConvex(hull))
# #False True
# #说明轮廓曲线是非凸的，凸包曲线是凸的
# img_contour= cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)

# 边界矩形
# x, y, w, h = cv2.boundingRect(contours[0])
# img_contour = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
# # 最小矩形
# rect = cv2.minAreaRect(contours[0])
# box = cv2.boxPoints(rect)
# box = np.int64(box)
# img_contour = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
# # 最小外切圆
# (x, y), radius = cv2.minEnclosingCircle(contours[0])
# center = (int(x), int(y))
# radius = int(radius)
# img_contour = cv2.circle(img, center, radius, (255, 0, 0), 2)
# #椭圆
# ellipse = cv2.fitEllipse(contours[0])
# cv2.ellipse(img, ellipse, (255, 255, 255), 2)
# # 直线拟合
# h, w, _ = img.shape
# # [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
# # lefty = int((-x * vy / vx) + y)
# # righty = int(((w - x) * vy / vx) + y)
# # cv2.line(img, (w - 1, righty), (0, lefty), (255, 0, 0), 2)
# output = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
#
# k = output[1] / output[0]
# b = output[3] - k * output[2]
# lefty = int(-k*output[2]+output[3])
# righty = int(((w - x) * k) + y)
# cv2.line(img, (w - 1, righty), (0, lefty), (255, 0, 0), 2)

# cv2.imshow("img_contour", img_contour)
#
# cv2.waitKey(0)

# img = cv2.imread(r"./image/1.jpg")
# img = cv2.pyrDown(img)
# img = cv2.pyrUp(img)
# cv2.imshow("img",img)
# cv2.waitKey(0)



# def pyramid_demo(image):
#     level = 3
#     temp = image.copy()
#     pyramid_images = []
#     for i in range(level):
#         dst = cv2.pyrDown(temp)  #先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
#         pyramid_images.append(dst)
#         cv2.imshow("pyramid_demo_%s"%i,dst)
#         temp = dst.copy()
#     return pyramid_images
# #
# def lapalian_demo(image):
#     pyramid_images = pyramid_demo(image) #拉普拉斯需要用到高斯金字塔结果
#
#     level = len(pyramid_images)
#     for i in range(level-1,-1,-1):  #从后向前2,1,0
#         if i == 0:
#             expand = cv2.pyrUp(pyramid_images[i]) #先上采样
#             lapls = cv2.subtract(image, expand) #使用高斯金字塔上一个减去当前上采样获取的结果，才是拉普拉斯金字塔
#             print(expand.shape)
#             print(lapls.shape)
#         else:
#             expand = cv2.pyrUp(pyramid_images[i])
#
#             lapls = cv2.subtract(pyramid_images[i-1],expand)
#
#         lapls = cv2.convertScaleAbs(lapls, alpha=5, beta=0)
#         cv2.imshow("lapls_down_%s"%i,lapls)
#
#
# src = cv2.imread(r"./image/1.jpg")  #读取图片
# src = cv2.resize(src,(600,600))
# cv2.namedWindow("input image",cv2.WINDOW_AUTOSIZE)    #创建GUI窗口,形式为自适应
# cv2.imshow("input image",src)    #通过名字将图像和窗口联系
# lapalian_demo(src)
# cv2.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
# cv2.destroyAllWindows()  #销毁所有窗口



# class Apple():
#     def __init__(self,image):
#         self.image = image
#
#     def pyramid_demo(self):
#
#         temp = self.image.copy()
#         pyramid_images = [self.image]
#         for i in range(7):
#             dst = cv2.pyrDown(temp)  # 先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
#             pyramid_images.append(dst)
#             temp = dst.copy()
#         return pyramid_images
#
#     def lapalian_demo(self):
#         pyramid_images = self.pyramid_demo()  # 拉普拉斯需要用到高斯金字塔结果
#         pyramid_image = [pyramid_images[6]]
#         level = len(pyramid_images)
#
#         for i in range(6, 0, -1):  # 从后向前6,5 ..,0
#             expand = cv2.pyrUp(pyramid_images[i])
#             lapls = cv2.subtract(pyramid_images[i - 1], expand)
#             pyramid_image.append(lapls)
#             lapls = cv2.convertScaleAbs(lapls, alpha=10, beta=0)
#             # cv2.imshow("lapls_down_%s" % i, lapls)
#         return pyramid_image
# src1 = cv2.imread(r"./image/26.jpg")  #读取图片
# src2 = cv2.imread(r"./image/27.jpg")  #读取图片
# src1 = cv2.resize(src1,(960,960))
# src2 = cv2.resize(src2,(960,960))
# apple1 = Apple(src1)
# apple1.pyramid_demo()
# lpA = apple1.lapalian_demo()
# apple2 = Apple(src2)
# apple2.pyramid_demo()
# lpB = apple2.lapalian_demo()
#
# LS = []
# for la, lb in zip(lpA, lpB):
#     rows, cols, dpt = la.shape
#     ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
#     LS.append(ls)
# # now reconstruct
#
# ls_ = LS[0]
# for i in range(1, 7):
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])
#     # cv2.imshow(f"ls_{i}",ls_)
# # image with direct connecting each half
# real = np.hstack((src1[:, :cols // 2], src2[:, cols // 2:]))
# img1=cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
# img2 = cv2.cvtColor(ls_,cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.subplot(121)
# plt.imshow(img1)
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img2)
# plt.axis('off')
# plt.show()

