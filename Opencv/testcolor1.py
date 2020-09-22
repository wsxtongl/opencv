import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#转灰度
# src = cv2.imread("1.jpg")
# dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# cv2.imshow("src show",dst)
# cv2.waitKey(0)
# 通道分离
# src = cv2.imread("1.jpg")
# src[...,1] = 0
# src[...,2] = 0
# cv2.imshow("src show",src)
# cv2.waitKey(0)

# # 提取图像中的字
# def rgb2hsv(r, g, b):
#     '''rgb转hsv'''
#     r, g, b = r/255.0, g/255.0, b/255.0
#     mx = max(r, g, b)
#     mn = min(r, g, b)
#     df = mx-mn
#     if mx == mn:
#         h = 0
#     elif mx == r:
#         h = (60 * ((g-b)/df) + 360) % 360
#     elif mx == g:
#         h = (60 * ((b-r)/df) + 120) % 360
#     elif mx == b:
#         h = (60 * ((r-g)/df) + 240) % 360
#     if mx == 0:
#         s = 0
#     else:
#         s = df/mx
#     v = mx
#     H = h / 2
#     S = s * 255.0
#     V = v * 255.0
#     return H, S, V
#

# img = Image.open('./image/11.jpg')
# # 懒加载模式
# pix = img.load()

# width = img.size[0]
# height = img.size[1]
# for x in range(width):
#     for y in range(height):
#         r, g, b = pix[x, y]
#         h,s,v = rgb2hsv(r,g,b)
#         if 0 < h < 10 or 156 < h < 180:
#             pix[x, y] = 0,0,0
#         else:
#             pix[x, y] = 255,255,255
#
# plt.imshow(img)
# plt.show()

# 提取图像中的字
# img = cv2.imread(r"./image/11.jpg")
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_blue = np.array([156, 0, 0])
# upper_blue = np.array([180, 255, 200])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# mask_inv = cv2.bitwise_not(mask)
# res = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('frame', img)
# cv2.imshow('mask', mask)
# cv2.imshow("res",res)
#
# cv2.waitKey(0)


# #直线
# img = cv2.imread("./image/1.jpg")
# line = cv2.line(img,(100,30),(210,400),color=(0,0,255),thickness=2)
# cv2.imshow("line",line)

# #圆
# circle = cv2.circle(img,(100,100),50,color=(0,0,255),thickness=2)
# cv2.imshow("circle",circle)


# #矩形
# rectangle = cv2.rectangle(img,(100,30),(210,180),color=(0,0,255),thickness=2)
# cv2.imshow("rectangle",rectangle)
# #椭圆  -1反向填充
# ellipse = cv2.ellipse(img, (100, 100), (100, 50), 0, 0, 360, (255, 0, 0),thickness = -1)
# cv2.imshow("rectangle",ellipse)
# cv2.waitKey(0)
# img = cv2.imread(r"image/1.jpg")
# pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]], np.int32)
# # 顶点个数：4，矩阵变成4*1*2维
# print(pts.shape)
# #isClosed是否闭合线段
# #polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
# cv2.polylines(img, [pts], True, (0, 255, 255), 2)
# cv2.imshow("pic show", img)
# cv2.waitKey(0)

# img = cv2.imread(r"./image/1.jpg")
# font = cv2.FONT_HERSHEY_SIMPLEX
# #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细，字体类型
# img = cv2.putText(img, 'beautiful girl', (10, 100), font, 2, (0, 0, 255), 1,)
# lineType=cv2.LINE_AA)
# cv2.imshow("pic show",img)
# cv2.waitKey(0)



# img = cv2.imread('./image/12.jpg')
# cv2.imshow('original_img', img)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# colorless_hsv = img_hsv.copy()
# #饱和度
# colorless_hsv[:, :, 1] = 0.8 * colorless_hsv[:, :, 1]
# colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
#
# #亮度
# darker_hsv = img_hsv.copy()
# colorless_hsv[:, :, 2] = 0.9 * colorless_hsv[:, :, 2]
# darker_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
# # cv2.imwrite('./out/darker.jpg', darker_img)
#
# # cv2.imwrite('./image/colorless.jpg', colorless_img)
# # cv2.imshow('dst', colorless_img)
# cv2.imshow("darker",darker_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
