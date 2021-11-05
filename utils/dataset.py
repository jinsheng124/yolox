import random
import numpy as np
from torch.utils.data import Dataset
from utils.clip import get_random_data
import math
from utils.utils import xyxya2corner,draw_rec
import cv2
class listDataset(Dataset):
    def __init__(self,
                 root,
                 patch = 4,
                 shape=None,
                 train=False):

        with open(root, 'r') as file:
            self.lines = file.readlines()

        if train:
            self.patch = int(self.lines[0].rstrip().split(':')[1])
            self.lines = self.lines[1:]
            random.shuffle(self.lines)
        else:
            self.patch = patch
        self.nSamples = len(self.lines)
        self.train = train
        self.shape = shape
    def __len__(self):
        return self.nSamples
    def xyxy2normal(self,y,normal = False):
        boxes = np.array(y[:, 1:5], dtype=np.float32)
        boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
        boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2
        if normal:
            _scale = np.array([self.shape[0],self.shape[1],self.shape[0],self.shape[1]])
            boxes/=_scale
            boxes = np.maximum(np.minimum(boxes, 1), 0)
        angle = y[:,5:6]*180/math.pi+90
        # angle = y[:,5:6]
        y = np.concatenate([boxes, angle, y[:, :1]], axis=-1)
        return y
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        line = self.lines[index].rstrip()
        if self.train:
            img,y = get_random_data(line,sep=self.patch,input_shape=self.shape,is_training=self.train)
            # if(len(y)):
            #     img = img.astype(np.uint8)
            #     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            #     boxes = xyxya2corner(y[:,1:6]).astype(np.int)
            #     for box in boxes:
            #         draw_rec(img,box)
            #     cv2.imshow("img",img)
            #     cv2.waitKey(0)
            if len(y) != 0:
                y = self.xyxy2normal(y)
            label = np.array(y, dtype=np.float32)
            img = img.transpose((2, 0, 1))/255.0
            return img, label
        else:
            path,img,image_size = get_random_data(line, sep = self.patch,input_shape=self.shape, is_training=self.train)
            img = img.transpose((0, 3, 1, 2))/255.0
            return path, img,image_size



def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.stack(images)
    return images, bboxes
def dataset_collate_val(batch):
    images = []
    path = []
    image_size = []
    for pa,img, sz in batch:
        images.append(img)
        path.append(pa)
        image_size.append(sz)
    images = np.concatenate(images,axis=0)
    return path,images,image_size