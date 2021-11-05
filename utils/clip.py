
from PIL import Image
import numpy as np
import cv2
from utils.utils import letterbox_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
def get_random_data(annotation_line, input_shape, sep=4,jitter=.3, hue=.1, sat=1.5, val=1.5,is_training = False):
    """实时数据增强的随机预处理"""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    w, h = input_shape
    start = 2 if line[1].isnumeric() else 1
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[start:]])
    if is_training:
        c_w = iw//sep
        c_h = ih//sep
        c_x = int(line[1])%sep*c_w
        c_y = int(line[1])//sep*c_h
        image = image.crop((c_x, c_y, c_x + c_w, c_y + c_h))
        iw = c_w
        ih = c_h
    else:
        crop_images = []
        stride_w = iw//sep
        stride_h = ih//sep
        nx = ny = sep
        for j in range(ny):
            sy = j * stride_h
            for i in range(nx):
                sx = i*stride_w
                tmp_img = np.array(image.crop((sx,sy,sx+stride_w,sy+stride_h)).resize((w, h), Image.BICUBIC))
                crop_images.append(tmp_img)
        crop_images = np.stack(crop_images,axis=0)
        # box[:,[1,3]] = box[:,[1,3]]/iw*w
        # box[:,[2,4]] = box[:,[2,4]]/ih*h
        return line[0],crop_images,[iw,ih]
    # 调整图片大小
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.7, 1.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    # 放置图片
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h),
                          (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 是否翻转图片
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域变换
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue*360
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:,:, 0]>360, 0] = 360
    x[:, :, 1:][x[:, :, 1:]>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

    # 调整目标框坐标
    box_data = np.zeros((len(box), 6))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [1, 3]] = box[:, [1, 3]] * nw / iw + dx
        box[:, [2, 4]] = box[:, [2, 4]] * nh / ih + dy

        mask = np.where(np.abs(box[:,5])<=1.57)[0]
        box[mask, 5] = np.arctan(np.tan(box[mask,5]) * nh/ih * iw/nw)
        if flip:
            box[:, [1, 3]] = w - box[:, [3, 1]]
            box[:,5] = -box[:,5]
        box[:, 1:3][box[:, 1:3] < 0] = 0
        box[:, 3][box[:, 3] > w] = w
        box[:, 4][box[:, 4] > h] = h
        box_w = box[:, 3] - box[:, 1]
        box_h = box[:, 4] - box[:, 2]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
        box_data = np.zeros((len(box), 6))
        box_data[:len(box)] = box
    if len(box) == 0:
        return image_data, []

    if (box_data[:, :4] > 0).any():
        return image_data, box_data
    else:
        return image_data, []
