'''
Description: 
Author: Jesse
Date: 2023-03-22 22:31:43
LastEditors: Jesse
LastEditTime: 2023-03-23 21:16:22
'''
from typing import List

import logging

import numpy as np

from common.camera import Camera
from common.face import Face
from common.face_model import MODEL3D
from common.face_landmark_estimator import LandmarkEstimator


logger = logging.getLogger(__name__)


class FaceEstimator:

    def __init__(self, args):
        self._args = args

        # 创建Camera类，类属性由‘calib/sample_params.yaml’给定
        self.camera = Camera(args.camera_params_path)

        # 人脸关键点检测
        self._landmark_estimator = LandmarkEstimator(args)



    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_face(self, image: np.ndarray, face: Face) -> None:
        # MODEL3D是一个FaceModel对象实例
        # 得到此时的人脸的位姿(旋转矩阵和平移向量)
        MODEL3D.estimate_head_pose(face, self.camera)
        # 得到此时检测到的人脸的68个关键点的三维坐标(相机坐标系下)
        MODEL3D.compute_3d_pose(face)
        # 得到面部和眼睛中心的坐标
        MODEL3D.compute_face_eye_centers(face)


