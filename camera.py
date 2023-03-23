import cv2
from yolov5_detect import Detect
import numpy as np
from facedetector import FaceDetector

class camera:
    def __init__(self,cascade_mode):
        # 初始化分类器
        self.lEyeParam=[0,0,0,0]
        self.rEyeParam=[0,0,0,0]
        self.lEyeImg = np.zeros([20, 20, 1], np.uint8)
        self.rEyeImg = np.zeros([20, 20, 1], np.uint8)
        self.myFaceDetector = FaceDetector()

        self.cascade_mode = cascade_mode
        if cascade_mode == 'eye':
            self.cascade=cv2.CascadeClassifier(r'net/haarcascade_eye.xml')
        elif cascade_mode == 'glass_eye':
            self.cascade=cv2.CascadeClassifier(r'net/haarcascade_eye_tree_eyeglasses.xml')
        elif cascade_mode == 'yolov5_eye':
            self.cascade = Detect()
        else:
            self.cascade=cv2.CascadeClassifier(r'net/haarcascade_eye.xml')

    def start_camera(self,index):
        # 根据index选择相机
        self.cap = cv2.VideoCapture(index)
    
    def stop_camera(self):
        # 根据index选择相机
        self.cap.release()

    def capture(self)->bool:
        # 捕获一帧图像
        success, self.frame = self.cap.read()
        if not success:
            return success
        
        self.frame=cv2.flip(self.frame,1)
        return success

    def detectEye(self)->bool:
        '''
        检测眼睛位置,如果只检测到一只或者没有检测到，则返回False,成功检测到两只眼睛则返回True
        '''
        gray = self.frame.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.cascade_mode =='yolov5_eye':
            eyes = self.cascade.get_eye_roi(self.frame)
        else:
            eyes = self.cascade.detectMultiScale(gray,1.1,3,minSize=(80, 80),maxSize=(150, 150))

        
        if len(eyes)!=2:
            return False
        if abs(eyes[0][1]-eyes[1][1])>100 or abs(eyes[0][0]-eyes[1][0])<80:
            return False
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(self.frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # 同时检测到两个眼睛
        lEyeParam = eyes[0] if eyes[0][0]<eyes[1][0] else eyes[1]
        self.lEyeImg = gray[lEyeParam[1]:lEyeParam[1]+lEyeParam[3],lEyeParam[0]:lEyeParam[0]+lEyeParam[2]]
        self.lEyeParam=lEyeParam
        rEyeParam = eyes[0] if eyes[0][0]>eyes[1][0] else eyes[1]
        self.rEyeImg = gray[rEyeParam[1]:rEyeParam[1]+rEyeParam[3],rEyeParam[0]:rEyeParam[0]+rEyeParam[2]]
        self.rEyeParam=rEyeParam
        return True
    
    def detectFace(self)->bool:
        self.frame =self.myFaceDetector.getFace(self.frame)
        return True
    