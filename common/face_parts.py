from typing import Optional

import enum

import numpy as np
from scipy.spatial.transform import Rotation


class FacePartsName(enum.Enum):
    FACE = enum.auto()
    REYE = enum.auto()
    LEYE = enum.auto()


class FaceParts:
    def __init__(self, name: FacePartsName):
        self.name = name
        self.center: Optional[np.ndarray] = None
        self.head_pose_rot: Optional[Rotation] = None
        self.normalizing_rot: Optional[Rotation] = None
        self.normalized_head_rot2d: Optional[np.ndarray] = None
        self.normalized_image: Optional[np.ndarray] = None

        self.normalized_gaze_angles: Optional[np.ndarray] = None
        self.normalized_gaze_vector: Optional[np.ndarray] = None
        self.gaze_vector: Optional[np.ndarray] = None

    # 返回距离
    @property
    def distance(self) -> float:
        return np.linalg.norm(self.center)

    # 二维角度变换为三维向量，归一化的注视角度转换为归一化的三维注视向量
    def angle_to_vector(self) -> None:
        # pitch, yaw = self.normalized_gaze_angles
        yaw, pitch = self.normalized_gaze_angles
        self.normalized_gaze_vector = np.array([
            -np.cos(pitch) * np.sin(yaw),
            -np.sin(pitch),
            -np.cos(pitch) * np.cos(yaw)
        ])

    # 反归一化注视向量，即
    def denormalize_gaze_vector(self, normalized_distance) -> None:
        # 归一化旋转矩阵 R
        normalizing_rot = self.normalizing_rot.as_matrix()
        # 缩放矩阵 S
        scale = np.array([
            [1, 0, 0],
            [0, 1, 0],
            # normalized_distance为归一化距离
            [0, 0, normalized_distance / self.distance],
        ],
            dtype=np.float)
        # 转换矩阵M = SR
        conversion_matrix = scale @ normalizing_rot
        # 反归一化 gr = gn @ R
        # self.gaze_vector = self.normalized_gaze_vector @ normalizing_rot
        
        # 反归一化 gr = M-1 @ R
        self.gaze_vector = np.linalg.inv(conversion_matrix) @ self.normalized_gaze_vector

    # (x,y,z)->(pitch,yaw)
    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        assert vector.shape == (3, )
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        return np.array([yaw, pitch])
