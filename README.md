该项目用于旋转目标检测，目标检测框架yolox。
旋转目标检测相比于目标检测，除了要预测类别，中心坐标x,y,宽高w,h五元组外，还需要预测矩形框的旋转角度a。

*1,项目结构*：

	|-train.py 训练
		|-训练参数
			|-multitrain 多尺度训练
			|-smooth_label 标签平滑
			|-amp 是否开启混合精度
			|-Train_Next 断点继续训练
			|-freeze,freeze_epoch 冻结训练参数
			|-adam 是否使用adam优化器
			|-patch,是否划分网格训练，预测时就将图片划分为patch*patch网格，分别检测并堆叠结果，适合超大图片
			|-lr 学习率，SGD初始学习率默认0.0001
		|-训练得到的权重和训练结果可视化会保存在创建的logs文件夹中，每次训练都会计算mAP值
	|-test.py 计算mAP值
	|-detect.py 预测单张图片，运行detect.py，输入图片路径即可进行预测。

	|-pretrained
		|-内有yolox_s预训练权重,若无权重文件，请注释掉train.py的166-171行，freeze设置为False,不冻结训练。
	|-model_data
		|-classes.txt文件，可以自定义类别训练自己的模型
	|-dataset
		|-generate.py 根据训练数据生成trainlist.txt,vallist.txt用于训练,可以根据自己数据集结构自己写代码，生成txt的每一行格式为：
			|-图片路径 类别1,left1,top1,right1,bottom1,angle1 类别2，left2,top2,right2,bottom2,angle2 ...(角度值为弧度，-pi/2~pi/2之间)
			|-generate.py中的patch参数，用来处理大图片但小bbox情况,会将图片划分为patchxpatch的网格，每一个网格单独看做一张图片，增大bbox面积比，可以有效提升mAP
			
		|-train 存在训练的标注图片和标签（图片和标签除后缀要同名）
		|-val 存放测试用的图片和标签，标签一般为矩形4角点坐标加类别
	|-utils
		|- clip.py 数据预处理,包括数据增强
		|-dataset.py 生成dataset
		|-region_loss.py yolox损失（含角度损失），角度损失参考CSL论文
		|-utils 工具包，涵盖解码、非极大抑制、iou,坐标转换等函数
	|-yolox
		|- darknet.py cspdarknet网络
		|- yolox.py yolox模型
		
*2，环境配置"
		torch>=1.6
		tqmd=4.55.1
		cuda>=10.1
		numpy=1.19.3
		opencv-python>=3.4.2.16
		opencv-contrib-python>=3.4.2.16
		matplotlib
		json
		Pillow
		shapely
	
*3,实际检测效果图(yolox_s轻量化网络)*

![result](https://github.com/jinsheng124/yolox/blob/main/dataset/val/result.jpg)
