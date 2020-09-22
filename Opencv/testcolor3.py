import cv2
import numpy as np

photo_img = cv2.imread("./image/01.jpg")
back_img = cv2.imread("./image/22.jpg")

rows,cols,ch = photo_img.shape
back_rows , back_cols ,back_ch = back_img.shape

# 透视转换计算
pts1 = np.float32([[0,0],[0,rows],[cols,0],[cols,rows]])
pts2 = np.float32([[207,185],[207,490],[617,220],[610,468]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(photo_img,M,(back_cols,back_rows))
cv2.imshow('dst',dst)
# 制作掩模 Mak
gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
ret,mask =cv2.threshold(gray_dst,0,255,cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img_bg = cv2.bitwise_and(back_img,back_img,mask=mask_inv)

res = cv2.add(img_bg,dst)


cv2.imshow("back_img",back_img)
cv2.imshow("res",res)
cv2.imshow("img_bg",img_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()