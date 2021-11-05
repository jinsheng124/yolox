import numpy as np
import math
from shapely.geometry import Polygon
import torch.nn as nn
import torch
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
    keypoint = [int(cx - w/2.0),int(cy - h/2.0),int(cx+w/2.0),int(cy+h/2.0),a]
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
def gaussian_label(label, num_class, u=0, sig=4.0):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    return np.concatenate([y_sig[math.ceil(num_class / 2) - int(label.item()):],
                           y_sig[:math.ceil(num_class / 2) - int(label.item())]], axis=0)
if __name__=="__main__":
    # 转换格式
    #将长边放在第三位，短边放在第4位，这样的角度更加符合直觉
    # classid,x_c,y_c,longside,shortside
    # po = [
    #             [
    #                 3532.7797967589277,
    #                 2830.1922922713475
    #             ],
    #             [
    #                 3513.3806027343203,
    #                 2798.0623771680916
    #             ],
    #             [
    #                 3542.7270651309564,
    #                 2780.3437583625755
    #             ],
    #             [
    #                 3562.1262591555637,
    #                 2812.4736734658313
    #             ]
    #         ]
    # poo = [
    #
    #             [
    #                 3513.3806027343203,
    #                 2798.0623771680916
    #             ],
    #             [
    #                 3542.7270651309564,
    #                 2780.3437583625755
    #             ],
    #             [
    #                 3562.1262591555637,
    #                 2812.4736734658313
    #             ],
    #             [
    #                 3532.7797967589277,
    #                 2830.1922922713475
    #             ]
    #         ]
    # k = poly_iou(po,poo)
    # print(k)
    # po_1 = corner2xyxya(po)
    # print(po_1)
    # po_1 = np.expand_dims(po_1,axis=0)
    # po_2 = xyxya2corner(po_1)
    # print(po_2[0])
    #角度损失计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BCEangle = nn.BCEWithLogitsLoss()
    #预测值
    ps = torch.randn(4,180)
    csl_label_flag = False
    if not csl_label_flag:
        #直接用交叉熵损失，即将角度分为180份，每一份都是一个预测值
        ttheta = torch.full_like(ps, 0)
        ttheta[range(4), 90] = 1
        langle_loss = BCEangle(ps, ttheta)  # BCE Θ类别损失以BCEWithLogitsLoss来计算
    else:
        ttheta = torch.zeros_like(ps)  # size(num, 180)
        for idx in range(len(ps)):  # idx start from 0 to len(ps)-1
            # 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor  tangle[i].shape=(num_i, 1)
            theta = torch.tensor(90.)  # 取出第i个layer中的第idx个目标的角度数值  例如取值θ=90
            # CSL论文中窗口半径为6效果最佳，过小无法学到角度信息，过大则角度预测偏差加大
            #此处用了高斯平滑
            csl_label = gaussian_label(theta, 180, u=0, sig=6)  # 用长度为1的θ值构建长度为180的csl_label
            ttheta[idx] = torch.from_numpy(csl_label)  # 将cls_label放入对应的目标中
        langle_loss = BCEangle(ps, ttheta)
    print(langle_loss)
