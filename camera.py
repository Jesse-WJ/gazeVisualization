import cv2
from yolov5_detect import Detect
import numpy as np

import argparse

from common.visualizer import Visualizer
from common.face_estimator import FaceEstimator

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='视线可视化.')
    
    # 人脸检测模型选择
    parser.add_argument('--detect_mode', type=str, default='dlib', help="使用哪种方式进行人脸检测", choices=['dlib'])
    parser.add_argument('--face_landmark_predictor_path', type=str, default='data/dlib/shape_predictor_68_face_landmarks.dat', help="人脸关键点预测器存放路径")
    
    # 相机参数设置
    parser.add_argument('--camera_params_path', type=str, default='data/calib/sample_params_my.yaml', help="相机参数存放路径")  
    
    # 运行demo文件可选参数
    parser.add_argument('--head_pose_axis_length', type=int, default=0.05, help="头部姿态轴的长度")
        
    args = parser.parse_args()
    return args

class camera:
    def __init__(self,cascade_mode):
        # 初始化分类器
        self.lEyeParam=[0,0,0,0]
        self.rEyeParam=[0,0,0,0]
        self.lEyeImg = np.zeros([20, 20, 1], np.uint8)
        self.rEyeImg = np.zeros([20, 20, 1], np.uint8)
        self.args = parse_args()
        self.face_estimator = FaceEstimator(self.args)
        self.visualizer = Visualizer(self.face_estimator.camera)

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
        
        # self.frame=cv2.flip(self.frame,1)
        return success

    def detectEye(self,faceimg,faceparam)->bool:
        '''
        检测眼睛位置,如果只检测到一只或者没有检测到，则返回False,成功检测到两只眼睛则返回True
        '''
        if len(faceimg)==0 or len(faceimg[0])==0:
            return False
        gray = faceimg.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.cascade_mode =='yolov5_eye':
            eyes = self.cascade.get_eye_roi(faceimg)
        else:
            eyes = self.cascade.detectMultiScale(gray,1.1,3,minSize=(80, 80),maxSize=(150, 150))

        
        if len(eyes)!=2:
            return False
        if abs(eyes[0][1]-eyes[1][1])>100 or abs(eyes[0][0]-eyes[1][0])<80:
            return False
        
        # 同时检测到两个眼睛
        lEyeParam = eyes[0] if eyes[0][0]<eyes[1][0] else eyes[1]
        self.lEyeImg = gray[lEyeParam[1]:lEyeParam[1]+lEyeParam[3],lEyeParam[0]:lEyeParam[0]+lEyeParam[2]]
        self.lEyeParam=lEyeParam
        self.lEyeParam[0]+=faceparam[0]
        self.lEyeParam[1]+=faceparam[1]

        rEyeParam = eyes[0] if eyes[0][0]>eyes[1][0] else eyes[1]
        self.rEyeImg = gray[rEyeParam[1]:rEyeParam[1]+rEyeParam[3],rEyeParam[0]:rEyeParam[0]+rEyeParam[2]]
        self.rEyeParam=rEyeParam
        self.rEyeParam[0]+=faceparam[0]
        self.rEyeParam[1]+=faceparam[1]
        
        
        cv2.rectangle(self.visualizer.image,[self.lEyeParam[0],self.lEyeParam[1]],[self.lEyeParam[0]+self.lEyeParam[2],self.lEyeParam[1]+self.lEyeParam[3]],(255,255,0),1)
        
        cv2.rectangle(self.visualizer.image,[self.rEyeParam[0],self.rEyeParam[1]],[self.rEyeParam[0]+self.rEyeParam[2],self.rEyeParam[1]+self.rEyeParam[3]],(255,255,0),1)


        
            

        return True
    
    def detectFace(self)->bool:
        undistorted = cv2.undistort(
            self.frame, self.face_estimator.camera.camera_matrix,
            self.face_estimator.camera.dist_coefficients)
        
        self.visualizer.set_image(self.frame.copy())

        faces = self.face_estimator.detect_faces(undistorted)
        if len(faces)<1:
            return False
        
        for face in faces:
            self.face_estimator.estimate_face(undistorted, face)

            self.faceCenter=face.center
            self.leyeCenter=face.leye.center
            self.reyeCenter=face.reye.center
            # faceParam = [x,y,w,h]
            self.faceParam = [int(face.bbox[0][0]),int(face.bbox[0][1]),int(face.bbox[1][0]-face.bbox[0][0]),int(face.bbox[1][1]-face.bbox[0][1])]
            self.faceImg = self.frame[int(face.bbox[0][1]):int(face.bbox[1][1]),int(face.bbox[0][0]):int(face.bbox[1][0])]

            self.detectEye(self.faceImg,self.faceParam)
            
            self.visualizer.draw_bbox(face.bbox)

            self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)
            
            length = self.args.head_pose_axis_length
            self.visualizer.draw_model_axes(face, length, lw=2)

        return True
    