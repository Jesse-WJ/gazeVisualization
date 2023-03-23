import torch
import numpy as np
import cv2

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device



class Detect:
    def __init__(
            self,
            weights='net/yolov5_eye.pt',  # model.pt path(s)
            data='net/yolov5_eye.yaml',  # dataset.yaml path
            imgsz=[640, 640],  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half

        self.device = select_device(device)
        self.model = DetectMultiBackend(weights,
                                        device=self.device,
                                        dnn=dnn,
                                        data=data)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.model.model.float()
        # self.model.warmup(imgsz=(1, 3, *imgsz),half=self.half)  # warmup
        
        

    def get_eye_roi(self,img0):
        self.im0 = img0.copy()
        img = letterbox(img0, self.imgsz, self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Run inference

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        result = []
        # Inference
        with torch.no_grad():
            pred = self.model(im)

        # NMS
            pred = non_max_suppression(pred,
                                    self.conf_thres,
                                    self.iou_thres,
                                    self.classes,
                                    self.agnostic_nms,
                                    max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for j, det in enumerate(pred):  # per image
            annotator = Annotator(self.im0,
                                  line_width=self.line_thickness,
                                  example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                          self.im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (
                        self.names[c]
                        if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                
                    if label.split(' ')[0] == 'eye' and float(label.split(' ')[1])>=0.6:
                        result.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2] - xyxy[0]),int(xyxy[3] - xyxy[1])])

            # Stream results
            self.im0 = annotator.result()
            # cv2.imshow('',self.im0)
            return result


    
    

