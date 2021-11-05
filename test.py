import torch
import numpy as np
import os
from utils.utils import nms,poly_iou,xyxya2corner,xywh2xyxy,decode_yolox_boxes
from tqdm import tqdm
import json
#计算Frame-mAP
def test(net,
         dataloader,
         class_names,
         epoch=0,
         critical_iou=0.5,
         conf_thres=0.05,
         nms_thres=0.4):
    bounding_boxes = []
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(devices)
    sep = dataloader.dataset.patch
    model_image_size = dataloader.dataset.shape
    for iteration,batch in enumerate(tqdm(dataloader)):
        idx, images,image_shapes = batch[0], batch[1],batch[2]
        with torch.no_grad():
            images = torch.from_numpy(images).float().to(devices)
            # labels = [torch.from_numpy(ann).float() for ann in labels]
            outputs = net(images)
            # loss_item = compute_loss(outputs, labels, np.reshape(anchors, [-1, 2]), len(class_names),label_smooth=0)
            # val_loss+=loss_item[0].item()
            output = decode_yolox_boxes(outputs,input_shape=model_image_size)
            #相对于608x608
            # [bs*num_img,anchors,17]
            output = xywh2xyxy(output)
            # [bs,num_img,anchors,17]
            output = output.reshape(-1,sep*sep,output.shape[1],output.shape[2])
            bscale =[]
            boffset = []
            for i in range(output.shape[0]):
                stride_w = image_shapes[i][0] // sep
                stride_h = image_shapes[i][1] // sep
                _scalew = stride_w / model_image_size[0]
                _scaleh = stride_h / model_image_size[1]
                offset = [[j%sep*stride_w,j//sep*stride_h]*2 for j in range(sep*sep)]
                bscale.append([_scalew, _scaleh] * 2)
                boffset.append(offset)
            bscale = torch.tensor(bscale).unsqueeze(1).unsqueeze(1).to(devices)
            boffset = torch.tensor(boffset).unsqueeze(2).to(devices)
            output[...,:4] = output[...,:4]*bscale+boffset
            #[bs,num_img*anchors,17]
            output = output.reshape(output.shape[0],-1,output.shape[3])
            # 非极大值抑制
            batch_detections = nms(output,conf_thres=conf_thres,nms_thres=nms_thres,only_objection=False,nms_link_classes=True,fast=False)
            # batch_detections = non_max_suppression(output,conf_thres=conf_thres,nms_thres=nms_thres,only_objection=False)

            for i, o in enumerate(batch_detections):
                #存在预测结果就保存，否则continue
                if o is None:
                    continue
                o = o.data.cpu()
                top_conf = np.array(o[:, 5])
                top_label = np.array(o[:, -1], np.int32)
                top_bboxes = np.array(o[:, :5])
                #截断，取整
                # top_bboxes = clip_coords(top_bboxes, image_shape)
                #boxes=np.round(boxes).astype('int32')
                for c, l, b in zip(top_conf, top_label, top_bboxes):
                    b = np.around(b, decimals=4)
                    bounding_boxes.append({
                        "conf": "%.6f" % c,
                        "class": int(l),
                        "bbox": b.tolist(),
                        "gt_path":idx[i]
                    })
    print("all boxes catched...")
    # 将所有结果按类别存成.json文件格式
    # [{"confident":0-1,"gt_box":坐标,"truthbox":txt_path,"class":0 or 1 or 2},...]
    current_dir = 'logs/detections/detections_' + str(epoch)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('logs/detections'):
        os.mkdir('logs/detections')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    for c in range(len(class_names)):
        #筛选出其中一类
        bounding_box = list(filter(lambda x: int(x["class"]) == c, bounding_boxes))
        if len(bounding_box) == 0:
            #如果未检测到该类，就从列表中剔除
            # count_class.remove(c)
            continue
        #按置信度排序，从大到小
        bounding_box.sort(key=lambda x: float(x['conf']), reverse=True)
        with open(current_dir +"/" +class_names[c] + "_dr.json", 'w') as outfile:
            json.dump(bounding_box, outfile,indent=1)
    print("all classes separated...")
    #所有真实框个数初始化
    label_dict={}
    t_class_list=[]
    #获得真实框
    lines = dataloader.dataset.lines
    for line in lines:
        li = line.rstrip().split()
        gt_box = np.array([np.array(list(map(float, box.split(',')))) for box in li[1:]])
        #特别注意
        # gt_box[:, 0] = gt_box[:, 0] - 1
        cur_class = list(map(int,gt_box[:,0].tolist()))
        t_class_list.extend(cur_class)
        label_dict[li[0]]=gt_box.tolist()
    #所有真实框个数
    truth_num_box = len(t_class_list)
    #所有预测框个数
    pre_num_box = len(bounding_boxes)
    print("num_truthbox: ",truth_num_box)
    print("num_prebox: ",pre_num_box)
    #真实框类别
    count_class = np.unique(t_class_list)
    #真阳性样本初始化
    tp = 0
    del bounding_boxes
    #计算ap
    Maplist = []
    for c in count_class:
        t_box_length = t_class_list.count(c)
        gt_class_path = current_dir +'/'+ class_names[c] + "_dr.json"
        if not os.path.exists(gt_class_path):
            Maplist.append(0.0)
            continue
        #打开刚刚保存的json,循环取出每一个框
        bounding_box = json.load(open(gt_class_path))
        #初始化真阳性序列
        pred_match = np.zeros(len(bounding_box))
        
        for i, obj in enumerate(bounding_box):
            #取出一个预测框
            pre_box = np.array(obj["bbox"])
            #读取真实框
            gt_box = np.array(label_dict[obj["gt_path"]])
            if len(gt_box) == 0:
                continue
            #------------------待更新-------------------------#
            # gt_box_t = torch.from_numpy(gt_box[:, 1:5]).float()
            # pre_box = pre_box.expand_as(gt_box_t)
            # overlaps = bbox_iou(pre_box, gt_box_t).numpy()
            #xywha2corner转换为角点
            #poly_iou计算overlaps
            #计算预测框与所有真实框iou
            bs = gt_box.shape[0]
            gt_box_t = xyxya2corner(gt_box[:,1:])
            pre_box = xyxya2corner(np.expand_dims(pre_box,0))
            overlaps = np.zeros(bs)
            for k in range(bs):
                overlaps[k] = poly_iou(gt_box_t[k],pre_box[0])
            #------------------------------------------------#
            #iou按从大到小排序
            sorted_ixs = np.argsort(-overlaps)
            for s in sorted_ixs:
                #依次判断iou是否大于0.5，小于则说明是假阳性样本，直接退出
                if overlaps[s] < critical_iou:
                    break
                #大于就判断预测类别和真实框类别是否一致，一致则真阳性，将序列位置置一，直接退出循环
                if obj["class"] == int(gt_box[s, 0]):
                    tp += 1
                    pred_match[i] = 1
                    #匹配到便去除真实框
                    label_dict[obj["gt_path"]].pop(s)
                    break
        #累加
        precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
        #此时召回率逐渐上升
        recalls = np.cumsum(pred_match).astype(np.float32) / t_box_length
        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        #保证准确率取每个召回率最大值
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])
        # Compute mean AP over recall range
        #取出召回率发生变化的点
        #为了求面积
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        Maplist.append(ap)
    aps={class_names[c]:round(Maplist[i],3) for i, c in enumerate(count_class)}
    mAP = sum(Maplist) / len(Maplist)
    recall = tp / truth_num_box
    precision = tp / pre_num_box if pre_num_box != 0 else 0.0
    return aps, mAP, recall, precision
if __name__ == "__main__":
    from utils.dataset import listDataset, dataset_collate_val
    from torch.utils.data import DataLoader
    from utils.utils import get_classes, get_anchors
    from yolox.yolox import YoloX
    import torch.nn as nn
    model_image_size = (640,640)
    batch_size = 1
    classes_path = 'model_data/yolo_classes.txt'
    model_path = 'logs/yolox_s_0.pt'
    phi = 's'
    #加载类别,先验框,模型,多GPU
    class_names = get_classes(classes_path)
    test_dataset = listDataset('dataset/vallist.txt',patch=4,shape = model_image_size,train=False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,
                                 num_workers=2,pin_memory=True,
                                 drop_last=False,collate_fn=dataset_collate_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(YoloX(len(class_names), phi=phi))
    net.load_state_dict(torch.load(model_path, map_location=device)["model"])
    net.to(device).eval()
    print("load model done!")
    APs, mAP, recall, precision = test(net,test_dataloader,class_names,epoch=0)
    print("each class ap:")
    print(str(APs)[1:-1])
    print("recall:{:.3f} precision:{:.3f} mAP:{:.3f}".format(recall, precision, mAP))
