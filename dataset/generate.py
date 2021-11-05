import os
import json
import math
from collections import defaultdict
import numpy
import glob
labels = ['A','B','C','D','E','F','G','H','I','J','K']
#生成训练集标签，格式
# 图片路径 框1 框2
# 框的格式 id,x1,y1,x2,y2,a
# 数据集解压到同级文件夹
#生成trainlist.txt,testlist.txt
def generate_dataset(path,save_name,patch = 1,train = True):
    img_name = list(filter(lambda x:x[-3:]=='png',os.listdir(path)))
    total_line = ""
    if train:
        total_line+=f"patch:{patch}\n"
    cur_labels = set()
    for iname in img_name:
        ip = path + '/'+iname
        ip_json = ip.split('.')[0]+'.json'
        data =json.load(open(ip_json,'r',encoding='utf-8'))
        root_path = "dataset/"+ip
        iw = data["imageHeight"]
        ih = data["imageWidth"]
        meshxy = defaultdict(list)
        boxes = []
        for k,shape in enumerate(data['shapes']):
            label = labels.index(shape['label'])
            cur_labels.add(shape['label'])
            points = shape['points']
            w = ((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2)**0.5
            h = ((points[1][0]-points[2][0])**2+(points[1][1]-points[2][1])**2)**0.5
            if w>h:
                if points[1][0] == points[0][0]:
                    a = math.pi / 2.0
                else:
                    a = math.atan((points[1][1] - points[0][1]) / (points[1][0] - points[0][0]))
            else:
                if points[1][0]  == points[2][0]:
                    a = math.pi / 2.0
                else:
                    a = math.atan((points[2][1] - points[1][1]) / (points[2][0] - points[1][0]))
                w,h = h,w
            cx = (points[0][0]+points[2][0])/2.0
            cy = (points[0][1]+points[2][1])/2.0
            a = round(a, 4)
            keypoint = [max(int(cx - w/2.0),0),max(int(cy - h/2.0),0),min(int(cx+w/2.0),iw),min(int(cy+h/2.0),ih)]
            if patch>1:
                sw = iw//patch
                sh = ih//patch
                s1 = keypoint[0]//sw+1
                e1 = math.ceil(keypoint[2]/sw)
                meshx = [keypoint[0]]+[i*sw for i in range(s1,e1)]+[keypoint[2]]
                s2 = keypoint[1]//sh+1
                e2 = math.ceil(keypoint[3]/sh)
                meshy = [keypoint[1]]+[i*sh for i in range(s2,e2)]+[keypoint[3]]
                for i in range(len(meshx)-1):
                    for j in range(len(meshy)-1):
                        num_grid = (meshy[j]//sh)*patch+(meshx[i]//sw)
                        line = str(label)+","+str(meshx[i]%sw)+","+str(meshy[j]%sh)+","+\
                               str((meshx[i+1]-1)%sw)+","+str((meshy[j+1]-1)%sh)+","+str(a)
                        meshxy[num_grid].append(line)
            else:
                line = str(label) + "," + ",".join(list(map(str,keypoint))) + "," + str(a)
                boxes.append(line)
        if patch>1:
            for key,value in meshxy.items():
                lines = root_path+" "+str(key)+" "+" ".join(value)+"\n"
                total_line+=lines
        else:
            lines = root_path+" "+" ".join(boxes)+"\n"
            total_line+=lines
    with open(save_name,"w") as f:
        f.write(total_line)
    print(sorted(list(cur_labels)))
if __name__=="__main__":
    generate_dataset('train','trainlist.txt',patch=4,train=True)
    generate_dataset('val', 'vallist.txt',patch=1,train=False)
