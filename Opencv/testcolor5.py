import cv2
import numpy as np
#透视变换
# img_bg = cv2.imread("./image/04.jpg")
# img_photo = cv2.imread("./image/colorless.jpg")
#
# bg_h,bg_w,bg_ch = img_bg.shape
# pho_h,pho_w,pho_ch = img_photo.shape
#
# pt1 = np.float32([[0,0],[0,pho_h],[pho_w,0],[pho_w,pho_h]])
# # pt2 = np.float32([[528,330],[528,456],[755,327],[756,455]])
# pt2 = np.float32([[557,227],[535,528],[1010,264],[1043,556]])
# M = cv2.getPerspectiveTransform(pt1,pt2)
# dst = cv2.warpPerspective(img_photo,M,(bg_w,bg_h))
# #制作掩膜
# gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# ret,mask = cv2.threshold(gray,2,255,cv2.THRESH_BINARY)
# Mask_inv = cv2.bitwise_not(mask)
# bg_img = cv2.bitwise_and(img_bg,img_bg,mask =Mask_inv)
#
# res = cv2.add(bg_img,dst)
#
# cv2.imshow("res",res)
# cv2.waitKey(0)

#膨胀操作
# src = cv2.imread("./image/6.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# dst = cv2.dilate(src,element)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
#腐蚀操作
# src = cv2.imread("./image/6.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# dst = cv2.erode(src,element)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
#开操作 先腐蚀再膨胀，去噪
# src = cv2.imread("./image/4.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# dst = cv2.morphologyEx(src,cv2.MORPH_OPEN,element,iterations=1)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)
# cv2.waitKey(0)
#闭操作，先膨胀再腐蚀，补漏洞
# src = cv2.imread("./image/4.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # iterations:迭代次数
# dst = cv2.morphologyEx(src,cv2.MORPH_CLOSE,element,iterations=1)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)
# cv2.waitKey(0)
#梯度操作，腐蚀减膨胀,镂空
# src = cv2.imread("./image/3.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # iterations:迭代次数
# dst = cv2.morphologyEx(src,cv2.MORPH_GRADIENT,element,iterations=1)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)
# cv2.waitKey(0)
#开运算图像-原图像，获取噪音
# src = cv2.imread("./image/4.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # iterations:迭代次数
# dst = cv2.morphologyEx(src,cv2.MORPH_TOPHAT,element,iterations=1)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)
# cv2.waitKey(0)
#闭运算图像 - 原图像,获取漏洞
# src = cv2.imread("./image/4.jpg")
# element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # iterations:迭代次数
# dst = cv2.morphologyEx(src,cv2.MORPH_BLACKHAT,element,iterations=1)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)
# cv2.waitKey(0)