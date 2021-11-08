该项目用于旋转目标检测，目标检测框架yolox,
预测值：
	类别，中心坐标x,y,宽高w,h,旋转角度a
	
|-train.py 训练
|-test.py 计算mAP值
|-detect.py 预测单张图片

	|----utils
		|-- clip.py 数据预处理
		|--dataset.py 生成dataset
		|--region_loss.py yolox损失（含角度损失）
		|--utils 工具包
|-yolox
	|-- darknet.py cspdarknet网络
	|-- yolox.py yolox模型
