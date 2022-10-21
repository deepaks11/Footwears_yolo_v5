import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import ( check_img_size, cv2, non_max_suppression,  scale_boxes, xyxy2xywh)
from utils.plots import Annotator,colors
from utils.augmentations import letterbox


weights = 'weights/shoes_slipper_4.pt'  # model path or triton URL
data = 'data/coco128.yaml'  # dataset.yaml path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt


class Footwear:

    def __init__(self,images,img0):

        self.images = images
        self.img0 = img0
        self.imgsz = (720, 720)  # inference size (height, width)
        self.conf_thres = 0.55  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.agnostic_nms = False
        self.hide_labels = False
        self.hide_conf = False

    def detect(self):


        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        frame = [letterbox(x, imgsz, stride)[0] for x in self.img0]
        frame = np.stack(frame, 0)
        frame = frame[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        frame = np.ascontiguousarray(frame)

        im = torch.from_numpy(frame).to(model.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, max_det=self.max_det)

        for i, det in enumerate(pred):  # per image
            s = '%g: ' % i
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(self.images.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            annotator = Annotator(self.images, line_width=self.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], self.images.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            final_img = annotator.result()

        return final_img




