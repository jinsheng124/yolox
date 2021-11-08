该项目用于旋转目标检测，目标检测框架yolox。
旋转目标检测相比于目标检测，除了要预测类别，中心坐标x,y,宽高w,h五元组外，还需要预测矩形框的旋转角度a。
项目结构如下：

	|-train.py 训练
	|-test.py 计算mAP值
	|-detect.py 预测单张图片

	|-utils
		|-- clip.py 数据预处理
		|--dataset.py 生成dataset
		|--region_loss.py yolox损失（含角度损失）
		|--utils 工具包
	|-yolox
		|-- darknet.py cspdarknet网络
		|-- yolox.py yolox模型
