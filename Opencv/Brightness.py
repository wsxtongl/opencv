import cv2 as cv
import numpy as np


def contrast_Ratio_brightness(arg):
    # arg参数：为接收新变量地址
    # a为对比度，g为亮度
    # cv.getTrackbarPos获取滑动条位置处的值
    # 第一个参数为滑动条1的名称，第二个参数为窗口的名称。
    a = cv.getTrackbarPos(trackbarName1, windowName)
    g = cv.getTrackbarPos(trackbarName2, windowName)
    h, w, c = image.shape
    mask = np.zeros([h, w, c], image.dtype)
    # cv.addWeighted函数对两张图片线性加权叠加
    dstImage = cv.addWeighted(image, a/10, mask, 1 - a, g)
    cv.imshow("dstImage", dstImage)


image = cv.imread(r'./image/1.jpg')
cv.imshow("Saber", image)
trackbarName1 = "Ratio_a"
trackbarName2 = "Bright_g"
windowName = "dstImage"
a = 5  # 设置a的初值。
g = 10  # 设置g的初值。
count1 = 40  # 设置a的最大值
count2 = 50  # 设置g的最大值
# 给滑动窗口命名，该步骤不能缺少！而且必须和需要显示的滑动条窗口名称一致。
cv.namedWindow(windowName)

# 第一个参数为滑动条名称，第二个参数为窗口名称，
# 第三个参数为滑动条参数，第四个为其最大值，第五个为需要调用的函数名称。
cv.createTrackbar(trackbarName1, windowName, a, count1, contrast_Ratio_brightness)
cv.createTrackbar(trackbarName2, windowName, g, count2, contrast_Ratio_brightness)
# 下面这步调用函数，也不能缺少。
contrast_Ratio_brightness(0)

cv.waitKey(0)
cv.destroyAllWindows()


