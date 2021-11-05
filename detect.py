import torch
import torch.nn as nn
import random
from utils.utils import get_classes, decode_yolox_boxes
from yolox.yolox import YoloX
from utils.utils import nms, bbox_iou,xywh2xyxy,xyxya2corner,clip_coords,draw_rec
import cv2
import numpy as np
import os
import json
class Detect_YOLOX(object):
    def __init__(self,
                 image_size=(640, 640),
                 model_path="logs/best.pt",
                 phi = "s",
                 classes_path='model_data/yolo_classes.txt',
                 conf_thres=0.5,
                 iou_thres=0.3,
                 sep_batch = 4,
                 is_savefile = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.class_names = get_classes(classes_path)
        self.net = nn.DataParallel(YoloX(len(self.class_names), phi = phi))
        self.net.load_state_dict(torch.load(model_path,map_location=self.device)["model"])
        self.net.to(self.device).eval()
        print("load model done!")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        self.sep_batch = sep_batch
        self.is_savefile = is_savefile
    def plot_one_box(self, box, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        draw_rec(img,box,color=color,thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cx = (box[0][0]+box[2][0])//2
            cy = (box[0][1]+box[2][1])//2
            c1 = (cx,cy-30)
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def detect(self, path):
        images = cv2.imread(path)

        image_name = path.split('/')[-1]
        ih,iw = images.shape[:2]
        # 加灰条，防失真，推荐
        crop_images = []
        stride_w = iw//self.sep_batch
        stride_h = ih//self.sep_batch
        nx = ny =self.sep_batch
        for j in range(ny):
            sy = j * stride_h
            for i in range(nx):
                sx = i*stride_w
                crop_image = cv2.resize(images[sy:sy+stride_w,sx:sx+stride_h],(self.image_size[0], self.image_size[1]),interpolation=cv2.INTER_AREA)
                crop_image = crop_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                crop_image = np.ascontiguousarray(crop_image)
                crop_images.append(crop_image)
        crop_images = np.stack(crop_images,axis=0)/255.0

        with torch.no_grad():
            crop_images = torch.from_numpy(crop_images).float().to(self.device)
            outputs = self.net(crop_images)
            output = decode_yolox_boxes(outputs,input_shape=self.image_size)
            #相对于608x608
            num_img = nx*ny
            _scalew = stride_w / self.image_size[0]
            _scaleh = stride_h / self.image_size[1]
            output = xywh2xyxy(output)
            # [num_img,anchors,17]
            scale = torch.tensor([_scalew, _scaleh, _scalew, _scaleh]).to(self.device)
            offset = torch.tensor([[i%nx*stride_w,i//ny*stride_h]*2 for i in range(num_img)]).unsqueeze(1).to(self.device)
            output[...,:4] = output[...,:4]*scale+offset
            #[num_img*anchors,17]
            output = output.reshape(1,-1,output.shape[2])
            # 非极大值抑制
            batch_detection = nms(output,conf_thres=self.conf_thres,nms_thres=self.iou_thres,only_objection=False,nms_link_classes=True,fast=False)[0]
            if batch_detection is None:
                return images
        batch_detection = batch_detection.data.cpu()
        top_conf = np.array(batch_detection[:, 5])
        top_label = np.array(batch_detection[:, -1], np.int32)
        top_bboxes = np.array(batch_detection[:, :5])
        # 截断，取整
        top_bboxes = clip_coords(top_bboxes, (iw,ih))
        print(top_bboxes[:,4])
        top_bboxes = xyxya2corner(top_bboxes)
        s = ""
        for c in np.unique(top_label):
            n = (top_label == c).sum()
            s += '%g %s, ' % (n, self.class_names[c])  # add to string
        if s:
            print("detected: ",s[:-2])
        if self.is_savefile:
            res = {}
            res["image_name"] = image_name
            res["labels"] = []
            for i,c in enumerate(top_label):
                each_label = {}
                each_label["category_id"] = self.class_names[c]
                each_label["points"] = top_bboxes[i].astype(np.float16).tolist()
                each_label["confidence"] = float(top_conf[i])
                res["labels"].append(each_label)
            if not os.path.exists("result"):
                os.mkdir("result")
            with open(rf"result/{image_name.split('.')[0]}.json","w",encoding='utf-8') as f:
                f.write(json.dumps(res,indent=1))
        for i, c in enumerate(top_label):
            label = '{} {:.2f}'.format(self.class_names[c], top_conf[i])
            box = top_bboxes[i].astype(np.int)
            self.plot_one_box(box, images, color=self.colors[c], label=label, line_thickness=3)
        return images
if __name__=="__main__":
    det = Detect_YOLOX(image_size=(640, 640),
                       model_path="logs/yolox_s_0.pt",
                       phi= 's',
                       classes_path='model_data/yolo_classes.txt',
                       conf_thres=0.3,
                       iou_thres=0.4,
                       sep_batch= 4,
                       is_savefile=False)
    while True:
        file_name = input("请输入图片路径：")
        if file_name =='q':
            break
        if not os.path.exists(file_name):
            continue
        img = det.detect(file_name)
        # img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()

