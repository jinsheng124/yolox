import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast,GradScaler

import numpy as np
from utils.dataset import listDataset, dataset_collate,dataset_collate_val
from utils.utils import get_classes,shuffle_net
from torch.utils.data import DataLoader
from yolox.yolox import YoloX
from test import test
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
import os
from utils.region_loss import YOLOLoss,weights_init
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_lr(optimizer,epoch,Epochs,init_lr=0.0001,milestones=[0.25,0.5,0.75,0.9],gamma=[0.5,0.2,0.5,0.2]):
    reduce = 1.0
    for i,m in enumerate(milestones):
        if epoch>=Epochs*m:
            if isinstance(gamma,list):
                reduce *= gamma[i]
            else:
                reduce *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*reduce
def draw_result(APs:dict):
    paras = {'figure.figsize': '10,10'}
    plt.rcParams.update(paras)
    plt.clf()
    xi = list(APs.keys())
    yi = list(APs.values())
    plt.bar(xi, yi, align="center", color="b", alpha=0.6)
    plt.xticks(xi, xi, rotation=60)
    for xn, yn in zip(xi, yi):
        plt.text(xn, yn + 0.01, "%.2f" % yn, ha="center", va="bottom", fontsize=10)
    plt.text(0, 1.1, f"mAP = {mAP:.3f}", fontsize=15)
    plt.ylim(0, 1)
    plt.ylabel("AP")
    plt.savefig("logs/results/best_AP.png")
# def get_optimizer(net,lr):
#     #采用不同学习率
#     _2d_param = list(map(id,net.module.c2d.backbone_2d.parameters()))
#     base_params = filter(lambda p: id(p) not in _2d_param,net.parameters())
#     optimizer = torch.optim.SGD([
#             {'params': base_params},
#             {'params': net.module.c2d.backbone_2d.parameters(), 'lr': lr*10}],
#             lr=lr, momentum=0.937,weight_decay=5e-4)
#     return optimizer

def fit_one_epoch(epoch, Epoch, gen, genval):
    total_loss,loss_conf,loss_cls,loss_loc = 0,0,0,0
    epoch_size = max(1, len(gen.dataset) // batch_size)
    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, labels = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).float().to(device)
                #-------多尺度训练,每10个batch--------#
                if multitrain:
                    if iteration % 10 ==0:
                        gz = 32
                        sl = random.uniform(0.7,1.5)
                        img_sz = [int(x*sl//gz*gz) for x in model_image_size]
                    images = F.interpolate(images,size=img_sz,mode='bilinear',align_corners=False)
                #-------------获得关键帧-------------#
                # labels = [torch.from_numpy(ann).float() for ann in labels]
                labels = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in labels]
            if not amp:
                optimizer.zero_grad()
                outputs = net(images)
                loss_item = loss_fn(outputs, labels)
                loss = loss_item[0]
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                with autocast():
                    outputs = net(images)
                    loss_item = loss_fn(outputs, labels)
                    loss = loss_item[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            #保留损失信息
            loss_conf += loss_item[1]
            loss_cls += loss_item[2]
            loss_loc += loss_item[3]
            total_loss += loss.item()
            del loss
            pbar.set_postfix(**{'loss': total_loss / (iteration + 1),
                    'lr': get_lr(optimizer)})
            pbar.update(1)
    train_loss = total_loss / (epoch_size + 1)
    conf_loss = loss_conf / (epoch_size + 1)
    cls_loss = loss_cls / (epoch_size + 1)
    loc_loss = loss_loc / (epoch_size + 1)
    # print('Start Validation')
    net.eval()
    APs, mAP, recall, precision = test(net,genval,class_names,epoch=epoch)
    print("each class ap:")
    print(str(APs)[1:-1])
    print("recall:{:.3f} precision:{:.3f} mAP:{:.3f}".format(recall, precision, mAP))
    return train_loss,conf_loss,cls_loss,loc_loss,APs, mAP, recall, precision

if __name__ == "__main__":
    #设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #超参数
    #--------------------------tricks--------------------------#
    #是否多尺度训练
    multitrain = False
    #是否标签平滑，可设为0-0.1的小值
    smooth_label = 0
    #混合精度
    amp = False
    #------------------------训练参数------------------------------#
    #是否恢复现场继续训练及其恢复权重路径
    Train_Next = False
    weight_path = "logs/last.pt"
    #总迭代次数
    start_epoch,end_epoch = 0,50
    #冻结次数，可选择关闭冻结
    freeze = True
    freeze_epoch = 5
    #学习率
    lr = 0.0001
    #是否使用Adam优化器
    adam = False
    # 模型size
    model_image_size = (640,640)
    #tensorboard
    use_tb_writer = False
    #除去最后一次训练的optimizer
    shuffle = True
    #-----------------------------------------加载数据集------------------------------------#
    batch_size = 4
    batch_size_val = 1
    patch = 4
    num_workers = 2
    train_dataset = listDataset('dataset/trainlist.txt',shape = model_image_size,train=True)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,
                                  pin_memory=True,shuffle=True,drop_last=True,
                                  collate_fn=dataset_collate)
    test_dataset = listDataset('dataset/vallist.txt',patch = patch,shape = model_image_size,train=False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size_val,
                                 num_workers=num_workers,pin_memory=True,
                                 drop_last=False,collate_fn=dataset_collate_val)

    #-------------------------加载类别----------------------------------------------------#
    classes_path = 'model_data/yolo_classes.txt'
    class_names = get_classes(classes_path)
    print(class_names)
    #----------------------------------------加载模型------------------------------------#
    phi = 's'
    torch.backends.cudnn.benchmark = True
    model = YoloX(len(class_names), phi = phi)
    weights_init(model)
    loss_fn = YOLOLoss(num_classes=len(class_names))
    model_path = 'pretrained/'+'yolox_'+phi+'.pth'
    print('Load weights {}.'.format(model_path))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # net = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(model,device_ids=[0])
    net = net.to(device)
    #-----------------------------------------------------------------------------------------#
    if amp:
        scaler = GradScaler()
    #冻结参数
    if freeze:
        for param in net.module.backbone.backbone.parameters():
            param.requires_grad = False
    #优化器
    if adam:
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.937,weight_decay=5e-4)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,25,45],gamma=0.1)
    #断点训练
    if Train_Next and os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        net.load_state_dict(checkpoint["model"])
        if checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_mAP = checkpoint["score"]
        del checkpoint
        torch.cuda.empty_cache()
        if start_epoch >= end_epoch:
            end_epoch += start_epoch
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('logs/results'):
        os.mkdir('logs/results')
    #-----------tensorboard----------------#
    if use_tb_writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir="logs/results")
        graph_inputs = torch.randn(1,3,*model_image_size).to(device)
        tb_writer.add_graph(model,(graph_inputs))
    print('Start training...')
    best_mAP = 0
    #--------------------------------------#
    #开始训练
    for epoch in range(start_epoch, end_epoch):
        if not adam:
            fit_lr(optimizer,epoch,Epochs=end_epoch-start_epoch,init_lr=lr)
        #判断是否解冻网络,并减少batch
        if epoch>= freeze_epoch and freeze:
            freeze = False
            for param in net.module.backbone.backbone.parameters():
                param.requires_grad = True
            del train_dataloader
            torch.cuda.empty_cache()
            batch_size = 4
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=2,
                                  pin_memory=True,shuffle=True,drop_last=True,collate_fn=dataset_collate)
        train_loss,conf_loss,cls_loss,loc_loss,APs, mAP, recall, precision = fit_one_epoch(epoch, end_epoch, train_dataloader, test_dataloader)
        # fit_one_epoch(epoch, end_epoch, train_dataloader, test_dataloader)
        #更新学习率
        # if lr_scheduler:
        #     lr_scheduler.step()
        #保存结果
        #-----用tensorboard保存结果以便可视化---------#
        if use_tb_writer:
            tags = ["train_loss","conf_loss","cls_loss","loc_loss","recall","precision","mAP@0.5"]
            for x,tag in zip([train_loss,conf_loss,cls_loss,loc_loss,recall, precision,mAP],tags):
                tb_writer.add_scalar(tag,x,epoch)
        #-------------------------------------------#
        if epoch == 0:
            s="{:<10s}"*8+"\n"
            s=s.format('epoch','loss','conf_loss','cls_loss','loc_loss','recall','precise','mAP')
            with open("logs/results/mAP.txt", "w") as f:
                f.write(s)
        s ="{:<10d}"+"{:<10.3f}"*7+"\n"
        s = s.format(epoch,train_loss,conf_loss,cls_loss,loc_loss,recall,precision,mAP)
        with open("logs/results/mAP.txt", "a+") as f:
            f.write(s)
        with open("logs/results/AP.txt", "a+") as f:
            ap = {"epoch": epoch, **APs}
            ap = str(ap) + "\n"
            f.write(ap)
        #保存模型
        state = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'score': mAP}
        print('Saving state, iter:', str(epoch + 1))
        torch.save(state, 'logs/last.pt')
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(state, 'logs/best.pt')
            draw_result(APs=APs)
    if shuffle:
        shuffle_net(model_path = 'logs/last.pt')
        shuffle_net(model_path = 'logs/best.pt')