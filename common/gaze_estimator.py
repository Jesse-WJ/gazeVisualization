from typing import List

import logging

import numpy as np
import torch

from common import Camera, Face, FacePartsName, MODEL3D
from head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from backbone.create_model import create_model

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, args):
        self._args = args

        # 创建Camera类，类属性由‘calib/sample_params.yaml’给定
        self.camera = Camera(args.camera_params_path)
        # 创建归一化相机
        self._normalized_camera = Camera(
            args.normalized_camera_params_path)
        # 人脸关键点检测
        self._landmark_estimator = LandmarkEstimator(args)
        # 头部姿态归一化
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            args.normalized_camera_distance)
        # 调用训练好的模型
        self._gaze_estimation_model = self._load_model()
        # 调用图像处理方法
        self._transform = args.transform

    def _load_model(self) -> torch.nn.Module:
        # 调用基础模型
        model = create_model(self._args.model_name)
        # 加载训练好的模型
        weights = torch.load(self._args.weights_path,
                                map_location='cpu')
        # 加载训练好的数据
        model.load_state_dict(weights['model'])
        model.to(torch.device(self._args.device))
        # 进入评估模式
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        # MODEL3D是一个FaceModel对象实例
        # 得到此时的人脸的位姿(旋转矩阵和平移向量)
        MODEL3D.estimate_head_pose(face, self.camera)
        # 得到此时检测到的人脸的68个关键点的三维坐标(相机坐标系下)
        MODEL3D.compute_3d_pose(face)
        # 得到面部和眼睛中心的坐标
        MODEL3D.compute_face_eye_centers(face)

        self._head_pose_normalizer.normalize(image, face)
        self._run_mpiifacegaze_model(face)

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            # 得到归一化后的图像
            image = eye.normalized_image
            # 归一化的头部姿态(pitch,yaw)
            normalized_head_pose = eye.normalized_head_rot2d
            # 右眼的头部姿态由左眼翻转得到
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            # 对图像使用与训练测试数据集一样的数据处理方法
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._args.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            predictions = self._gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._args.device)
        with torch.no_grad():
            image = image.to(device)
            prediction = self._gaze_estimation_model(image)
            prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector(self._args.normalized_camera_distance)
