import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common import Camera, FaceParts, FacePartsName


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera,
                 normalized_distance: float):
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance

    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        # 旋转矩阵R
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        # 利用 W = Cn @ S @ R @ Cr'对检测到的图像进行归一化处理，之后再对眼部图像进行直方图均衡化处理
        self._normalize_image(image, eye_or_face)
        # 计算归一化的头部姿态(pitch,yaw)
        self._normalize_head_pose(eye_or_face)

    def _normalize_image(self, image: np.ndarray,
                         eye_or_face: FaceParts) -> None:
        # 相机投影矩阵的逆Cr'
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)
        # 归一化相机的投影矩阵Cn
        normalized_camera_matrix = self.normalized_camera.camera_matrix
        # 缩放矩阵S
        scale = self._get_scale_matrix(eye_or_face.distance)
        # 转换矩阵M = SR
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()
        # 投影矩阵 W = CnMCr'
        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv
        # 对图像进行归一化处理
        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (self.normalized_camera.width, self.normalized_camera.height))

        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            # 转换为灰度图像
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            # 直方图均衡化
            normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image
        # eye_or_face.scale = scale


    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        # 归一化的头部旋转矩阵Rn = Rr * R
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        """
        # # 归一化的头部旋转矩阵Rn = R * Rr
        # normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        # # 归一化的头部旋转矩阵Rn = S * R * Rr
        # normalized_head_rot = eye_or_face.scale * eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        # # 归一化的头部旋转矩阵Rn = Rr * M'
        # normalized_head_rot = eye_or_face.head_pose_rot * (eye_or_face.scale * eye_or_face.normalizing_rot).T
        """
        # 得到对应的欧拉角
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        """返回旋转矩阵R"""
        # ravel()将center展平,然后再变成单位向量
        z_axis = _normalize_vector(center.ravel())  # 归一化相机坐标系的z轴zc
        head_rot = head_rot.as_matrix()     # 头部旋转矩阵(检测到的人脸坐标系到相机坐标系的旋转矩阵)
        head_x_axis = head_rot[:, 0]        # 头部坐标系的x轴
        y_axis = _normalize_vector(
            np.cross(z_axis, head_x_axis))       # 归一化相机坐标系的y轴
        x_axis = _normalize_vector(
            np.cross(y_axis, z_axis))            # 归一化相机坐标系的x轴
        return Rotation.from_matrix(
            np.vstack([x_axis, y_axis, z_axis]))        # 旋转矩阵R

    def _get_scale_matrix(self, distance: float) -> np.ndarray:     # 缩放矩阵S
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            # normalized_distance为归一化距离
            [0, 0, self.normalized_distance / distance],
        ],
            dtype=np.float)
