from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
import math
from shapely.geometry import Polygon
import cv2
def decode_yolox_boxes(outputs, input_shape):
    grids   = []
    strides = []
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    #---------------------------------------------------#
    # outputs[:,:,4:5] = torch.sigmoid(outputs[:,:,4])*3.142-1.571

    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    angle_pred = torch.argmax(outputs[:,:, 4:184],dim=-1,keepdim=True).float() - 90
    angle_pred = angle_pred/180*math.pi
    for h, w in hw:
        #---------------------------#
        #   根据特征层的高宽生成网格点
        #---------------------------#
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #---------------------------#
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    #---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #------------------------#
    #   根据网格点进行解码
    #------------------------#
    outputs[..., :2]    = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides
    #-----------------#
    #   归一化
    #-----------------#
    # outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
    # outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
    return torch.cat((outputs[...,:4],angle_pred,outputs[...,184:]),dim=-1)
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
def correct_boxes(boxes, model_image_size,image_shape):
    ap = boxes[:,4:5]
    boxes = boxes[:,:4]
    w,h = image_shape
    wi,hi=model_image_size
    boxes/=np.array([[wi,hi,wi,hi]])
    shape = np.array([w, h, w, h])
    offset = (shape - np.max(shape)) / np.max(shape) / 2.0
    offset = np.expand_dims(offset, axis=0)
    boxes = (boxes + offset) / (1.0 + 2 * offset) * np.expand_dims(shape, axis=0)
    boxes = np.concatenate((boxes,ap),axis=-1)
    return boxes
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0]=np.clip(boxes[:, 0],0, img_shape[0]) # x1
    boxes[:, 1]=np.clip(boxes[:, 1],0, img_shape[1])  # y1
    boxes[:, 2]=np.clip(boxes[:, 2],0, img_shape[0])  # x2
    boxes[:, 3]=np.clip(boxes[:, 3],0, img_shape[1])  # y2
    return boxes
def bbox_iou(box1, box2, x1y1x2y2=True,DIoU=False):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    if DIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + 1e-16
        rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        return iou - rho2 / c2  # DIoU

    return iou
def xywh2xyxy(x):
    y = x.new(x[...,:4].shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    x[...,:4] = y
    return x
def xyxya2corner(p):
    #batchsize*5
    #旋转矩阵
    bs = p.shape[0]
    corners = np.zeros((bs,4,2))
    for i in range(bs):
        w = p[i][2] - p[i][0]
        h = p[i][3] - p[i][1]
        set_x = (p[i][2] + p[i][0])/2.0
        set_y = (p[i][3] + p[i][1])/2.0
        poly_set = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
        a = p[i][4]
        a_mu = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
        poly_set = poly_set@a_mu
        poly_c = np.array([set_x,set_y])
        poly_c = np.expand_dims(poly_c,0).repeat(4,axis=0)
        corners[i] = poly_c + poly_set
    return corners
def corner2xyxya(points):
    w = ((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5
    h = ((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2) ** 0.5
    cx = (points[0][0] + points[2][0]) / 2.0
    cy = (points[0][1] + points[2][1]) / 2.0

    if points[1][0] - points[0][0] == 0:
        a = math.pi / 2.0
    else:
        a = math.atan((points[1][1] - points[0][1]) / (points[1][0] - points[0][0]))
    keypoint = [int(cx - w / 2.0), int(cy - h / 2.0), int(cx + w / 2.0), int(cy + h / 2.0),a]
    return np.array(keypoint)
def poly_iou(x1,x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x1 = Polygon(x1[:8].reshape((4,2)))
    x2 = Polygon(x2[:8].reshape((4, 2)))
    if not x1.is_valid or not x2.is_valid:
        return 0
    inter = Polygon(x1).intersection(Polygon(x2)).area
    union = x1.area + x2.area - inter
    if union==0:
        return 0
    else:
        return inter/union
def draw_rec(img,rec,color = (0,0,255),thickness = 2):
    cv2.line(img,(rec[0][0],rec[0][1]),(rec[1][0],rec[1][1]),color,thickness=thickness)
    cv2.line(img, (rec[1][0], rec[1][1]), (rec[2][0], rec[2][1]), color, thickness=thickness)
    cv2.line(img, (rec[2][0], rec[2][1]), (rec[3][0], rec[3][1]), color, thickness=thickness)
    cv2.line(img, (rec[3][0], rec[3][1]), (rec[0][0], rec[0][1]), color, thickness=thickness)
def nms_poly(dets, scores, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引  list
    """
    polys = xyxya2corner(dets)
    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = poly_iou(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep

def nms(prediction,conf_thres=0.5,nms_thres=0.6,only_objection=True,nms_link_classes = True,fast = False):
    #xyxya
    output = [None] * prediction.shape[0]
    max_wh = 4097
    xc = prediction[..., 5] > conf_thres
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[xc[image_i]]
        if not image_pred.size(0):
            continue
        if only_objection:
            box= image_pred[:, :5]
            conf=image_pred[:,5:6]
            j = torch.argmax(image_pred[:, 6:],dim=1,keepdim=True)
            image_pred = torch.cat((box, conf, j.float()), 1)
        else:
            image_pred[:, 6:] *= image_pred[:, 5:6]
            box= image_pred[:, :5]
            conf, j = image_pred[:, 6:].max(1, keepdim=True)
            image_pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # if sep_batch>1:
        #     #找比框中参数最大的那个，加1为偏移单位，乘以类别作为偏移量
        #     offset = image_pred[:,6:7] * max_wh if nms_link_classes else 0
        #     # offset = image_pred[:,5:6] * (box.max()-box.min()+1)
        #     #偏移不影响iou计算
        #     boxes, scores = image_pred[:, :4] + offset, image_pred[:, 5]
        #     image_preds = []
        #     each_batch = boxes.shape[0]//sep_batch
        #     for i in range(sep_batch):
        #         each_image_pred = image_pred[i*each_batch:(i+1)*each_batch]
        #         each_boxes = boxes[i*each_batch:(i+1)*each_batch]
        #         each_scores = scores[i*each_batch:(i+1)*each_batch]
        #         keep = torchvision.ops.nms(each_boxes, each_scores, nms_thres)
        #         image_preds.append(each_image_pred[keep])
        #     image_pred = torch.cat(image_preds,dim=0)
        if fast:
            offset = image_pred[:,6:7] * max_wh if nms_link_classes else 0
            # offset = image_pred[:,5:6] * (box.max()-box.min()+1)
            #偏移不影响iou计算
            boxes, scores = image_pred[:, :4] + offset, image_pred[:, 5]
            keep=torchvision.ops.nms(boxes, scores,nms_thres)
        else:
            offset = image_pred[:,6:7] * max_wh if nms_link_classes else 0
            boxes = torch.cat((image_pred[:, :4] + offset,image_pred[:,4:5]),dim=-1)
            scores = image_pred[:,5]
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            keep = nms_poly(boxes,scores,nms_thres)
        output[image_i] = image_pred[keep]

    return output

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]
def shuffle_net(model_path):
    net = torch.load(model_path)
    net["optimizer"] = None
    torch.save(net, model_path)