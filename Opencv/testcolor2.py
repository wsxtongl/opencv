import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
#像素超过阈值，赋黑色或白色，不超过相反
# img = cv2.imread("./image/1.jpg")
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#
# ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
# titles = ['img','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
# for i in range(0,6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
#
#     plt.title(titles[i])
#     plt.axis("off")
# plt.show()

# img1 = cv2.imread('./image/1.jpg')
# img2 = cv2.imread('./image/6.jpg')

# outputimage = cv2.bitwise_xor(img1, img2, )
# out = cv2.bitwise_or(img1, img2, )
# andout = cv2.bitwise_and(img1, img2, )
# cv2.imshow("xor", outputimage)
# cv2.imshow("or", out)
# cv2.imshow("and", andout)

# img1 = cv2.imread('./image/1.jpg')
# img2 = cv2.imread('./image/6.jpg')
# rows, cols, channels = img1.shape
# roi = img2[0:rows, 0:cols]
#
# img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, Mask = cv2.threshold(img1gray, 170, 255, cv2.THRESH_BINARY)
# Mask_inv = cv2.bitwise_not(Mask)
# img1_bg = cv2.bitwise_and(roi, roi, mask=Mask)
# img1_fg = cv2.bitwise_and(img1, img1, mask=Mask_inv)
# # img2_fg = cv2.bitwise_and(roi, roi, mask=Mask_inv)
# dst = cv2.add(img1_bg, img1_fg)
# img2[0:rows, 0:cols] = dst
# cv2.imshow('Mask', Mask)
# cv2.imshow('Mak_inv', Mask_inv)
# cv2.imshow("img1_bg",img1_bg)
#
# cv2.imshow("img1_fg",img1_fg)

# def callback(object):
#     pass

# cv2.namedWindow('image')
#
# [x, y, z] = img1.shape
# # 创建一个相同规格的图像，可以自己读取一张图用切片工具
# # 选出相同大小的矩阵
# img2 = np.zeros([x, y, z], img1.dtype)
# B, G, R = 10, 88, 21  # 自己调色
# img2[:, :, 0] = np.uint8(B)
# img2[:, :, 1] = np.uint8(G)
# img2[:, :, 2] = np.uint8(R)
# cv2.createTrackbar('alpha', 'image', 0, 100, callback)
#
# while True:
#     Alpha = cv2.getTrackbarPos('alpha', 'image')/100
#     img3 = cv2.addWeighted(img1, Alpha, img2, 1-Alpha, 0)
#     cv2.imshow('image', img3)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()

# img1 = cv2.imread("./image/1.jpg")
# rows,cols,channel = img1.shape
#
# #M = np.float32([[1,0,50],[0,1,50]])
# M = np.float32([[2,0,0],[0,2,0]])
# dst = cv2.warpAffine(img1, M, (cols, rows))
# cv2.imshow("dst",dst)
# cv2.waitKey(0)

src = cv2.imread('./image/1.jpg')
rows, cols, channel = src.shape
# M = np.float32([[1, 0, 50], [0, 1, 50]])
# M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# M = np.float32([[-1, 0, cols], [0, 1, 0]])
# M = np.float32([[1, 0.5, 0], [0, 1, 0]])
# M = np.float32([[1, 0, 0], [0, -1, rows]])
# M = np.float32([[-1, 0, cols], [0, -1, rows]])
# M1 = cv2.getRotationMatrix2D((0,0), 10, 1)
# M = np.float32([[1, 1, 0], [0, 1, 0]])
# dst = cv2.warpAffine(src, M, (cols, rows))
# M1 = cv2.warpAffine(src, M1, (cols, rows))
# cv2.imshow('src pic', src)
# cv2.imshow('dst pic', dst)
# cv2.imshow('M1',M1)
# cv2.waitKey(0)

import sys

img = cv2.imread('./image/1.jpg')
# cv2.imshow("original", img)

# 可选，扩展图像，保证内容不超出可视范围
img = cv2.copyMakeBorder(img, 300, 300, 300, 300, cv2.BORDER_CONSTANT, 0)
w, h = img.shape[0:2]

anglex = 0
angley = 30
anglez = 0  # 是旋转
fov = 42
r = 0


def rad(x):
    return x * np.pi / 180


def get_warpR():
    global anglex, angley, anglez, fov, w, h, r
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    return warpR


def control():
    global anglex, angley, anglez, fov, r

    # 键盘控制
    if 27 == c:  # Esc quit
        sys.exit()
    if c == ord('w'):
        anglex += 1
    if c == ord('s'):
        anglex -= 1
    if c == ord('a'):
        angley += 1
        print(angley)
        # dx=0
    if c == ord('d'):
        angley -= 1
    if c == ord('u'):
        anglez += 1
    if c == ord('p'):
        anglez -= 1
    if c == ord('t'):
        fov += 1
    if c == ord('r'):
        fov -= 1
    if c == ord(' '):
        anglex = angley = anglez = 0
    if c == ord('e'):
        print("======================================")
        print('Rotation Matrix:')
        print(r)
        print('angle alpha(anglex):')
        print(anglex)
        print('angle beta(angley):')
        print(angley)
        print('dz(anglez):')
        print(anglez)

def control1():
    global anglex, angley, anglez, fov, r

    # 键盘控制
    # if 27 == c:  # Esc quit
    #     sys.exit()
    for i in range(1):
        for j in range(1):
            anglex += 5
            anglez += 5
            time.sleep(0.01)
        for a in range(1):
            anglex -= 2
            anglez -= 2
            time.sleep(0.01)
        for b in range(1):
            angley += 4
            time.sleep(0.01)
        for e in range(1):
            angley -= 4
            time.sleep(0.01)




    # if c == ord('t'):
    #     fov += 1
    # if c == ord('r'):
    #     fov -= 1
    # if c == ord(' '):
    #     anglex = angley = anglez = 0
    # if c == ord('e'):
    #     print("======================================")
    #     print('Rotation Matrix:')
    #     print(r)
    #     print('angle alpha(anglex):')
    #     print(anglex)
    #     print('angle beta(angley):')
    #     print(angley)
    #     print('dz(anglez):')
    #     print(anglez)

while True:
    warpR = get_warpR()

    result = cv2.warpPerspective(img, warpR, (h, w))
    cv2.namedWindow('result', 2)
    cv2.imshow("result", result)
    c = cv2.waitKey(30)
    control1()





