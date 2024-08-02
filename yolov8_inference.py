from ultralytics import YOLO
import cv2
import os
import pandas as pd

# Load a model
model = YOLO('/home/mzm/project/yqh/yolov7/runs/detect/train4/weights/best.pt')  # pretrained YOLOv8n model

img_path = '/home/mzm/project/yqh/yolov7/datasets/kitchen/inference/'


file_list = []
label_list = []
for filename in os.listdir(img_path):
	file_list.append(filename)
	img = cv2.imread(img_path+filename)
	results = model(img)
	# View results
	for r in results:
		cls_list = r.boxes.cls.cpu().numpy().tolist()
		print(f"{filename}：{cls_list}")
		if len(cls_list) == 0:
			label = 0
		else:
			# 使用循环遍历列表
			for i in range(len(cls_list)):
				# 判断是否为需要替换的值
				if cls_list[i] == 0:
					# 替换为新的值 BIT 1 : 抽烟  1
				    cls_list[i] = 1
				elif cls_list[i] == 1:
					# 替换为新的值 BIT 2 : 赤膊  2
				    cls_list[i] = 2
				elif cls_list[i] == 2:
					# 替换为新的值 BIT 3 : 老鼠 4
				    cls_list[i] = 4
				elif cls_list[i] == 3:
					# 替换为新的值 BIT 4 : 猫 8
				    cls_list[i] = 8
				elif cls_list[i] == 4:
					# 替换为新的值 BIT 5 : 狗 16
				    cls_list[i] = 16
			label = sum(cls_list)
		label_list.append(label)

#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'filename':file_list,'result':label_list})
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("results.csv",index=False,sep=',')