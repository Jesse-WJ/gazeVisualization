'''
Description: 
Author: Jesse
Date: 2023-03-21 20:31:35
LastEditors: Jesse
LastEditTime: 2023-03-22 20:35:47
'''
from typing import List

import dlib
import numpy as np
# import yacs.config

from common.face import Face


class LandmarkEstimator:
    def __init__(self, args):
        # 人脸检测模式
        self.mode = args.detect_mode
        if self.mode == 'dlib':
            # 人脸检测器
            self.detector = dlib.get_frontal_face_detector()
            # 人脸关键点预测器
            self.predictor = dlib.shape_predictor(args.face_landmark_predictor_path)
        else:
            raise ValueError

    # 检测人脸区域
    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        else:
            raise ValueError

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        # detector第一个参数表示图片，第二个参数表示是否将图像进行放大，0表示不放大，返回矩形[（x1,y1）（x2,y2）]
        # 人脸候选框集合bboxes
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected
