'''
Description: 
Author: Jesse
Date: 2023-03-22 19:50:47
LastEditors: Jesse
LastEditTime: 2023-03-22 22:07:07
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-21 17:18
# @Author  : yunshang
# @FileName: show_demo.py
# @Software: PyCharm
# @desc    : 用于可视化视线方向

from typing import Optional

import cv2
import logging
import pathlib
import datetime
import argparse
import numpy as np
from torchvision import transforms


from common import Face, Visualizer, GazeEstimator, FacePartsName
from utils import LinePlaneIntersection

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='使用AlexNet在MPIIFaceGaze上的预训练模型进行视线可视化.')

    # 视线估计模型相关参数
    parser.add_argument('--model_name', type=str, default='AlexNet', help="使用哪种网络模型进行视线估计")
    parser.add_argument('--num-output', default=2, type=int, help='视线估计输出的维度')
    parser.add_argument('--weights_path', type=str, default='model_p10_best.pth', help = '模型权重文件存放路径')
    parser.add_argument('--device', type=str, default='cpu', help="使用CPU还是GPU进行demo演示")
    parser.add_argument('--transform', help="模型对输入图像需要进行的图像预处理")
    
    # 人脸检测模型选择
    parser.add_argument('--detect_mode', type=str, default='dlib', help="使用哪种方式进行人脸检测", choices=['dlib'])
    parser.add_argument('--face_landmark_predictor_path', type=str, default='data/dlib/shape_predictor_68_face_landmarks.dat', help="人脸关键点预测器存放路径")
    
    # 相机参数设置
    parser.add_argument('--camera_params_path', type=str, default='data/calib/sample_params_my.yaml', help="相机参数存放路径")
    parser.add_argument('--normalized_camera_params_path', type=str, default='data/calib/normalized_camera_params_face.yaml', help='归一化相机参数存放路径')
    parser.add_argument('--normalized_camera_distance', type=float, default=0.6, help='归一化相机距离人眼中心的距离(米)')
    
    
    # 运行demo文件可选参数
    parser.add_argument('--use_camera', type=bool, default=True, help='是否使用相机进行实时视线估计')
    parser.add_argument('--video_path', type=str, default='', help="使用已有的视频进行视线估计")
    parser.add_argument('--display_on_screen', type=bool, default=True, help='是否在屏幕上显示')
    parser.add_argument('--wait_time', type=int, default=1, help="刷新图像的频率时间")
    parser.add_argument('--output_dir', type=str, default='', help="输出视频保存目录")
    parser.add_argument('--output_file_extension', type=str, default='mp4', help="输出视频的格式", choices=['mp4', 'flv'])
    parser.add_argument('--head_pose_axis_length', type=int, default=0.05, help="头部姿态轴的长度")
    parser.add_argument('--gaze_visualization_length', type=int, default=0.3, help="可视化视线的长度")
    parser.add_argument('--show_bbox', type=bool, default=True, help="是否显示人脸框")
    parser.add_argument('--show_head_pose', type=bool, default=True, help="是否显示头部姿态")
    parser.add_argument('--show_landmarks', type=bool, default=True, help="是否显示人脸关键点标记")
    parser.add_argument('--show_normalized_image', type=bool, default=False, help="是否显示标准化后的图片")
    parser.add_argument('--show_template_model', type=bool, default=False, help="是否显示模板模型")
    
    args = parser.parse_args()
    return args

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx*30, pos[1] + dy*30]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out
    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, args):
        self.args = args
        # 注视估计器
        self.gaze_estimator = GazeEstimator(args)
        self.visualizer = Visualizer(self.gaze_estimator.camera)
        # VideoCapture对象
        self.cap = self._create_capture()
        # 视频输出目录
        self.output_dir = self._create_output_dir()
        # 视频流写入对象
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.args.show_bbox
        self.show_head_pose = self.args.show_head_pose
        self.show_landmarks = self.args.show_landmarks
        self.show_normalized_image = self.args.show_normalized_image
        self.show_template_model = self.args.show_template_model
        self.file_path = "gaze_point1.txt"

    def run(self) -> None:
        while True:
            # 是否显示
            if self.args.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            # 读取视频帧
            ok, frame = self.cap.read()
            if not ok:
                break
            # cv2.imshow('',frame)
            # cv2.waitKey(0)
            # 校正因为相机本身镜头产生的畸变，返回校正后的相机拍摄的图像
            # 输入原图, 相机内参矩阵, 畸变系数
            undistorted = cv2.undistort(
                frame, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)

            self.visualizer.set_image(frame.copy())
            # 检测图像中的人脸，返回Face对象实例
            faces = self.gaze_estimator.detect_faces(undistorted)
            for face in faces:
                self.gaze_estimator.estimate_gaze(undistorted, face)
                self._draw_face_bbox(face)
                self._draw_head_pose(face)
                self._draw_landmarks(face)
                self._draw_face_template_model(face)
                self._draw_gaze_vector(face)
                self._display_normalized_image(face)

            if self.args.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            if self.writer:
                self.writer.write(self.visualizer.image)
            if self.args.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    # 创建捕获器
    def _create_capture(self) -> cv2.VideoCapture:
        # 使用摄像头
        if self.args.use_camera:
            cap = cv2.VideoCapture(1)
        # 使用已有的视频
        elif self.args.video_path:
            cap = cv2.VideoCapture(self.args.video_path)
        else:
            raise ValueError
        # 设置高和宽
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    # 创建视频输出目录
    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.args.output_dir:
            return
        output_dir = pathlib.Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    # 创建视频流写入对象
    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        # 输出文件格式
        ext = self.args.output_file_extension
        if ext == 'mp4':
            # 视频流写入格式
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    # 接收各种按键指令并改变各种标志
    def _wait_key(self) -> None:
        key = cv2.waitKey(self.args.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.args.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.args.display_on_screen:
            return
        if not self.show_normalized_image:
            return

        normalized = face.normalized_image

        if self.args.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.args.gaze_visualization_length
        with open(self.file_path, 'a+') as f:
            f.write(f"模型输出角度向量:{face.normalized_gaze_angles}\n \
            模型输出三维向量:{face.normalized_gaze_vector}\n \
            视线三维向量:{face.gaze_vector}\n \
            脸部中心三维向量:{face.center}\n \
            视线角度向量:{face.vector_to_angle(face.gaze_vector)}\n\n")
        print(face.center, '\t', face.gaze_vector)
        
        # point = LinePlaneIntersection(face.center, face.gaze_vector, np.array([0,0,0]), np.array([0,0,1]))
        self.visualizer.draw_3d_arrowed_line(face.center, face.center + length * face.gaze_vector)
        # print(face.center)
        # self.visualizer.draw_3d_points(face.center.reshape(-1, 3))
        # print(face.gaze_vector)
        # for key in [FacePartsName.REYE, FacePartsName.REYE]:
        #     eye = getattr(face, key.name.lower())
        #     # self.visualizer.draw_3d_line(
        #     # eye.center, eye.center + length * face.gaze_vector)
        #     self.visualizer.draw_3d_arrowed_line(face.center, face.center + length * face.gaze_vector)
        #
        #     point = LinePlaneIntersection(eye.center, face.gaze_vector, np.array([0,0,0]), np.array([0,0,1]))
            # print(point)
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        # logger.info(f'[face] point: {face.gaze_vector[0]*100}')
        
        

def main(args):
    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((448, 448)),
                                         transforms.ToTensor()])
    args.transform = data_transform
    
    demo = Demo(args)
    demo.run()
    
    pass

if __name__ == '__main__':
    args = parse_args()
    
    main(args)
    # model = get_model(num_output=args.num_output)
    # weights = torch.load(args.weights_path, map_location='cpu')
    # model.load_state_dict(weights, strict=False)
    #
    # detector = RetinaFace()
    # # cap = cv2.VideoCapture(cam)
    #
    # # Check if the webcam is opened correctly
    # if not cap.isOpened():
    #     raise IOError("Cannot open webcam")
    #
    # with torch.no_grad():
    #     while True:
    #         success, frame = cap.read()
    #         start_fps = time.time()
    #
    #         faces = detector(frame)
    #         if faces is not None:
    #             for box, landmarks, score in faces:
    #                 if score < .95:
    #                     continue
    #                 x_min = int(box[0])
    #                 if x_min < 0:
    #                     x_min = 0
    #                 y_min = int(box[1])
    #                 if y_min < 0:
    #                     y_min = 0
    #                 x_max = int(box[2])
    #                 y_max = int(box[3])
    #                 bbox_width = x_max - x_min
    #                 bbox_height = y_max - y_min
    #                 # x_min = max(0,x_min-int(0.2*bbox_height))
    #                 # y_min = max(0,y_min-int(0.2*bbox_width))
    #                 # x_max = x_max+int(0.2*bbox_height)
    #                 # y_max = y_max+int(0.2*bbox_width)
    #                 # bbox_width = x_max - x_min
    #                 # bbox_height = y_max - y_min
    #
    #                 # Crop image
    #                 img = frame[y_min:y_max, x_min:x_max]
    #                 img = cv2.resize(img, (224, 224))
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 im_pil = Image.fromarray(img)
    #                 img = data_transform(im_pil)
    #                 # img = Variable(img).cuda(gpu)
    #                 img = img.unsqueeze(0)
    #
    #                 # gaze prediction
    #                 gaze = model(img)
    #                 gaze_pitch = gaze[0][0]
    #                 gaze_yaw = gaze[0][1]
    #                 # (gaze_pitch, gaze_yaw) = model(img)
    #
    #                 draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (gaze_pitch, gaze_yaw),
    #                           color=(0, 0, 255))
    #                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    #         myFPS = 1.0 / (time.time() - start_fps)
    #         cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
    #                     cv2.LINE_AA)
    #
    #         cv2.imshow("Demo", frame)
    #         if cv2.waitKey(1) & 0xFF == 27:
    #             break
    #         success, frame = cap.read()
    
    
    