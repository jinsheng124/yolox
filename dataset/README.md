1、train、val文件夹下放图片(.png)和标签(.json)
2、标注格式见json文件，需要包括旋转矩形4角点坐标，类别信息
3、generate.py可以再同级目录生成trainlist.txt和vallist.txt用于训练和验证
4、trainlist.txt格式：
	图片路径 patch(如果patch大于1，会将图片拆分） 类别、x1,y1,x2,y2,a（弧度）
   testlist.txt无拆分参数patch
