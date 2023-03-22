import numpy as np
import math
import json
import pandas as pd
from scipy.optimize import fsolve
from math import cos,sin,pi
from scipy.optimize import leastsq

def save_csv(arr, csv_filename=None):
    """Save the data in csv format"""
    if csv_filename is None:
        csv_filename = "csv.csv"
    arr_df = pd.DataFrame(arr)
    arr_df.to_csv(csv_filename, float_format='%.3f', index=False, header=False)

def LinePlaneIntersection(p1, v1, p2, v2):
    '''
    求空间直线和空间平面的交点
    p1和v1为空间直线经过的点和方向向量
    p2和v2为空间平面经过的点和法向量
    '''
    # Point = p1+t*v1
    # np.dot(Point,v2)=np.dot(p2,v2)
    t = (np.dot(p2, v2)-np.dot(p1, v2))/np.dot(v1, v2)
    return p1+t*v1


def LineLineIntersection(p1, v1, p2, v2):
    '''
    求空间两直线的交点
    v1和v2是方向向量
    p1和p2是直线上一点
    '''
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    startPointSeg = p2-p1
    vecS1 = np.cross(v1, v2)
    vecS2 = np.cross(startPointSeg, v2)
    num = np.dot(vecS2, vecS1)/(np.linalg.norm(vecS1)**2)

    return p1+v1*num


def Get_angle(v1, v2):
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    return math.acos(np.dot(v1, v2))/math.pi*180


def Save_json(info, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(info, f)


# 求线球交点 取靠近相机的交点
def line_sphere_intersection(line_p, line_v, center, r):
    """
    求解线球交点
    :param line_p: 直线上一点
    :param line_v: 直线方向向量
    :param center: 球面中心
    :param r: 球面半径
    :return: 两个交点(本应用为一个交点)
    """

    point1 = np.zeros((3,), dtype=np.float64)
    point2 = np.zeros((3,), dtype=np.float64)

    p1 = line_p[0]
    p2 = line_p[1]
    p3 = line_p[2]
    v1 = line_v[0]
    v2 = line_v[1]
    v3 = line_v[2]
    x0 = center[0]
    y0 = center[1]
    z0 = center[2]

    a = (math.pow(v1, 2) + math.pow(v2, 2) + math.pow(v3, 2)) / math.pow(v1, 2)
    b = 2 * (-x0 + (p2 - v2 / v1 * p1) * v2 / v1 - y0 * v2 /
             v1 + (p3 - v3 / v1 * p1) * v3 / v1 - z0 * v3 / v1)
    c = math.pow(x0, 2) + math.pow(p2 - v2 / v1 * p1, 2) - 2 * y0 * (p2 - v2 / v1 * p1) + math.pow(y0, 2) + \
        math.pow(p3 - v3 / v1 * p1, 2) - 2 * z0 * \
        (p3 - v3 / v1 * p1) + math.pow(z0, 2) - math.pow(r, 2)
    deta = math.pow(b, 2) - 4 * a * c

    if deta < 0:
        return np.array(([math.nan, math.nan, math.nan]))
    else:
        x1 = (-b + math.sqrt(deta)) / 2 / a
        x2 = (-b - math.sqrt(deta)) / 2 / a
        y1 = v2 * x1 / v1 + p2 - v2 / v1 * p1
        y2 = v2 * x2 / v1 + p2 - v2 / v1 * p1
        z1 = v3 * x1 / v1 + p3 - v3 / v1 * p1
        z2 = v3 * x2 / v1 + p3 - v3 / v1 * p1
        # point1取值范数最小的那个向量
        if np.linalg.norm(np.array(([x1, y1, z1]))) < np.linalg.norm(np.array(([x2, y2, z2]))):
            point1 = np.array(([x1, y1, z1]))
            point2 = np.array(([x2, y2, z2]))
        else:
            point2 = np.array(([x1, y1, z1]))
            point1 = np.array(([x2, y2, z2]))

    return point1


def get_R(Sx,Sy,Sz,Dx,Dy,Dz):
    '''
    求原坐标系到目标坐标系的变换矩阵
    '''

    def func(x):
        a=x[0]
        b=x[1]
        c=x[2]

        Rx=np.array(([1,0,0],[0,cos(a),sin(a)],[0,-sin(a),cos(a)]))
        Ry=np.array(([cos(b),0,-sin(b)],[0,1,0],[sin(b),0,cos(b)]))
        Rz=np.array(([cos(c),sin(c),0],[-sin(c),cos(c),0],[0,0,1]))

        R=np.dot(np.dot(Rx,Ry),Rz)


        error1 = np.dot(R,Sx)-Dx
        error2 = np.dot(R,Sy)-Dy
        error3 = np.dot(R,Sz)-Dz

        # return error1,error2,error3
        return [error1[0],error1[1],error1[2],error2[0],error2[1],error2[2],error3[0],error3[1],error3[2]]

    init_x = np.random.uniform(0,pi*2,3)
    root = leastsq(func, init_x)
    root = root[0]
    # root = fsolve(func,init_x)

    a=root[0]
    b=root[1]
    c=root[2]
    Rx=np.array(([1,0,0],[0,cos(a),sin(a)],[0,-sin(a),cos(a)]))
    Ry=np.array(([cos(b),0,-sin(b)],[0,1,0],[sin(b),0,cos(b)]))
    Rz=np.array(([cos(c),sin(c),0],[-sin(c),cos(c),0],[0,0,1]))

    R=np.dot(np.dot(Rx,Ry),Rz)

    return R
   
