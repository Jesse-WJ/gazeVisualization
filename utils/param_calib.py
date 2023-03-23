import cv2
from math import sqrt

def ellipse_calib(img,threshold):
    """
    瞳孔分割函数
    :parameter img1：输入为一张灰度图片
    :return temp1:返回参数包含瞳孔中心，外切矩形的长宽和瞳孔拟合椭圆的偏角
    """
    img1 = img.copy()

    # 二值化 第二个参数阈值可调  可多次修改，找到最佳值
    ret, th = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    erosion = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('', erosion)
    # cv2.waitKey(0)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num = len(contours)
    temp1 = [[0,0],[0,0],0]
    if num < 2:  # 轮廓的数量小于2个，说明前面的图像处理有问题，需要调节阈值
        return 0
    for n in range(num):
        count = len(contours[n])
        if count < 6:  # 轮廓的像素个数大于6才能拟合椭圆
            continue
        # 拟合椭圆
        rotatedRect = cv2.fitEllipseAMS(contours[n])
        if (rotatedRect[1][0]*rotatedRect[1][1])>200:
        # if 1.3 > (rotatedRect[1][0]) / (rotatedRect[1][1]) > 0.7:  # 瞳孔近似为一个圆形 通过椭圆外切矩形的长宽比筛选
            temp1 = rotatedRect
    return temp1


def SegmentSpotClib(img,threshold,ellipse_parameter):
    """
    分割亮斑函数
    :param img2: 输入为一张灰度图片
    :return: 返回左右两个亮斑的位置
    """
    img2 = img.copy()
    # 二值化
    ret, th = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow('', th)
    # cv2.waitKey(0)
    # 形态学处理
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # erosion = cv2.morphologyEx(th, cv2.MORPH_ERODE, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # erosion = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    erosion = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('', erosion)
    # cv2.waitKey(0)
    tempCenter = []
    tempCenterY = []
    tempCenter2 = []

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 2:
        for i in range(len(contours)):
            if len(contours[i]) > 2:
                # 拟合光斑的外切矩形
                # 由于光斑的像素比较少，没法用拟合椭圆的方法
                tempRect = cv2.boundingRect(contours[i])
                # tempRect = cv2.minAreaRect(contours[i])
                # if abs(tempRect[0][0] - pupilcenter[0]) < 100:
                # print(tempRect[0] + tempRect[2] / 2, tempRect[1] + tempRect[3] / 2)
                # print(tempRect)
                M_OO = 0
                M_O1 = 0
                M_10 = 0
                for ii in range(tempRect[2]):
                    a = []
                    for jj in range(tempRect[3]):
                        x = int(tempRect[0]) + ii
                        y = int(tempRect[1]) + jj
                        a.append(erosion[y, x])
                        M_OO = M_OO + erosion[y, x]
                        M_O1 = M_O1 + x * erosion[y, x]
                        M_10 = M_10 + y * erosion[y, x]
                    # print(a)
                X = M_O1 / M_OO
                Y = M_10 / M_OO
                # cv2.rectangle(erosion, (tempRect[0], tempRect[1]),
                #               (tempRect[0] + tempRect[2], tempRect[1] + tempRect[3]), (255, 255, 255))
                # cv2.imshow('', erosion)
                # cv2.waitKey(0)
                # if 1.2 > (tempRect[2] / tempRect[3]) > 0.8:
                #     # 还是用长宽比进行筛选
                if sqrt(pow(ellipse_parameter[0][0]-X,2)+pow(ellipse_parameter[0][1]-Y,2))<30:
                    tempCenter.append([X, Y])
    return tempCenter

def Imshow(img,roi_param,ellipse_parameter,SpotParameter):
    # 把瞳孔和光斑圈出来
    ellipse=ellipse_parameter
    if not isinstance(ellipse_parameter,int) and ellipse_parameter[0][0]!=0:
        # 把瞳孔中心标出来
        # print(ellipse_parameter,roi_param)
        # ellipse_parameter[0][0]+=roi_param[0]
        # ellipse_parameter[0][1]+=roi_param[1]
        ellipse=[[ellipse_parameter[0][0]+roi_param[0],ellipse_parameter[0][1]+roi_param[1]],[ellipse_parameter[1][0],ellipse_parameter[1][1]],ellipse_parameter[2]]
        cv2.ellipse(img, ellipse, (255, 0, 0),-1)
        cv2.line(img, (int(ellipse[0][0]) - 5, int(ellipse[0][1])),
                    (int(ellipse[0][0]) + 5, int(ellipse[0][1])), (255, 255, 255), )
        cv2.line(img, (int(ellipse[0][0]), int(ellipse[0][1] - 5)),
                    (int(ellipse[0][0]), int(ellipse[0][1] + 5)), (255, 255, 255), )

    if len(SpotParameter)>0:
        # 把瞳孔和光斑圈出来
        for i in range(len(SpotParameter)):
            SpotParameter[i][0]+=roi_param[0]
            SpotParameter[i][1]+=roi_param[1]
            cv2.ellipse(img, (SpotParameter[i], (7, 7), 0), (0, 0, 255),-1)

    
    return img,ellipse,SpotParameter
