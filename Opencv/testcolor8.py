# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# image = cv2.imread("./image/1.jpg")
# image = cv2.GaussianBlur(image, (5, 5), 50)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray", image_gray)
# edges = cv2.Canny(image_gray, 100, 150)
# cv2.imshow("image_edges", edges)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b)) # 直线起点横坐标
#     y1 = int(y0 + 1000 * (a)) # 直线起点纵坐标
#     x2 = int(x0 - 1000 * (-b)) # 直线终点横坐标
#     y2 = int(y0 - 1000 * (a)) # 直线终点纵坐标
#
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv2.waitKey(0)
#
# image = cv2.imread("./image/28.jpg")
# dst = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#
# dst = cv2.Canny(dst, 90, 150)
# cv2.imshow('dst',dst)
# circle = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=40, param2=33,
# minRadius=20, maxRadius=200)
# if not circle is None:
#     circle = np.uint16(np.around(circle))
# for i in circle[0, :]:
#     cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
# cv2.imshow("circle", image)
# cv2.waitKey(0)


# img = cv2.imread('./image/30.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )
# cv2.imshow('thresh',thresh)
# # noise removal
# # kernel = np.ones((3, 3), np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# cv2.imshow("open",opening)
# #膨胀操作
# sure_bg = cv2.dilate(opening, kernel, iterations=7)
# dist_transform = cv2.distanceTransform(opening,3, 5)
# print(dist_transform.max())
# ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# cv2.imshow('fg',sure_fg)
#
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# cv2.imshow('unknown',unknown)
# # Marker labelling
# ret, markers1 = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers1 + 1
#
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
#
# markers3 = cv2.watershed(img, markers)
#
# img[markers3 == -1] = [0,0,255]
# cv2.imshow("img", img)
# cv2.waitKey(0)