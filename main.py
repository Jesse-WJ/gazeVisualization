import cv2
import logging
import argparse

from common.visualizer import Visualizer
from common.gaze_estimator import GazeEstimator

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = parse_args()

    cap = cv2.VideoCapture(0)
    
    gaze_estimator = GazeEstimator(args)
    visualizer = Visualizer(gaze_estimator.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_estimator.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_estimator.camera.height)


    while 1:
        ok, frame = cap.read()

        if not ok:
            break

        undistorted = cv2.undistort(
            frame, gaze_estimator.camera.camera_matrix,
            gaze_estimator.camera.dist_coefficients)
        
        visualizer.set_image(frame.copy())

        faces = gaze_estimator.detect_faces(undistorted)
        for face in faces:
            gaze_estimator.estimate_gaze(undistorted, face)
            visualizer.draw_bbox(face.bbox)

            visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)
            
            length = args.head_pose_axis_length
            visualizer.draw_model_axes(face, length, lw=2)

            euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
            pitch, yaw, roll = face.change_coordinate_system(euler_angles)
            logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

            visualizer.image = visualizer.image[:, ::-1]
            cv2.imshow('frame', visualizer.image)
            cv2.waitKey(10)

