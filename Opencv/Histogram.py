import cv2
import matplotlib.pyplot as plt
import numpy as np
# def hisEqulColor(img):
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     channels = cv2.split(ycrcb)
#     cv2.equalizeHist(channels[0],channels[0])
#     cv2.merge(channels, ycrcb)
#     cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR,img)
#     return img
#
# im = cv2.imread('./image/12.jpg')
# cv2.imshow('im1', im)
#
#
# eq = hisEqulColor(im)
# cv2.imshow('image2',eq )
# cv2.waitKey(0)
# 2D直方图
# img = cv2.imread('./image/1.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# cv2.imshow("hist",hist)
# cv2.waitKey(0)
#抠图
img1 = cv2.imread(r'./image/9.jpg')
img2 = cv2.imread(r'./image/10.jpg')
img2_hsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
img1_hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([img2_hsv],[0,1],None,[180,255],[0,180,0,255])
hist = cv2.normalize(hist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([img1_hsv],[0,1],hist,[0,180,0,255],1)
elment = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
cv2.filter2D(dst,-1,elment)
# dst = cv2.GaussianBlur(dst,(5,5),1)
# dst = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,elment,iterations=1)

ret,thresh = cv2.threshold(dst,0,255,cv2.THRESH_BINARY)
# thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,elment,iterations=1)
# thresh = cv2.erode(thresh,elment)
thresh1 = cv2.GaussianBlur(thresh, (3, 3), 5)
thresh2 = cv2.addWeighted(thresh, 3, thresh1, -1, 0)
#二值图转三通道
thresh2 = cv2.merge((thresh,thresh,thresh))
# cv2.imshow('thresh2',thresh2)
res = cv2.bitwise_and(img1,thresh2)
res = np.hstack((img1,thresh2,res))
cv2.imshow("gauss",thresh1)
cv2.imshow("dst",res)
cv2.waitKey(0)
