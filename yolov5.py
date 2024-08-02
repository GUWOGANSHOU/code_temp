pip install -r requirements.txt

python train.py --img 640 --batch 50 --epochs 20 --data ./data/tobacco.yaml --weights ./weights/yolov5s.pt --nosave --cache


# tobacco.yaml
path: /root/yolov5//datasets/tobacco
train: # train images (relative to 'path')  16551 images
  - images/train
val: # val images (relative to 'path')  4952 images
  - images/val
test: # test images (optional)

# Classes
names:
  0: business license
  1: tobacco license
  2: warning board

python detect.py --weights /root/yolov5/runs/train/exp3/weights/best.pt --img 640 --conf 0.25 --source /root/yolov5/datasets/tobacco/images/val


# 推理输出
import os

def process_yolo_txts(txt_dir, output_file):
    """
    处理YOLOv5的输出.txt文件，转换格式并保存到新文件。
    :param txt_dir: 包含.txt文件的目录
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w') as out_f:
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_dir, filename)
                with open(file_path, 'r') as in_f:
                    for line in in_f:
                        data = line.strip().split()
                        class_index = int(data[0])
                        x, y, w, h = map(float, data[1:5])
                        confidence = float(data[5])
                        class_name = class_names[class_index]
                        out_f.write(f"{x},{y},{w},{h},{confidence},{class_name}\n")

# 示例调用
txt_dir = '/root/yolov5/runs/detect/exp4/labels'
output_file = '/root/yolov5/runs/detect/exp4/output.txt'

# 假设类别名称如下：
class_names = ['business_license', 'tobacco_license', 'warning_board']

process_yolo_txts(txt_dir, output_file)



# detect.py 中去掉/gn 就不会归一化 x y w h
# xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
# xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # no normalized xywh



# 将归一化的坐标转换为像素坐标，可以忽略
import os

def convert_normalized_to_pixel_coords(normalized_coords, img_width, img_height):
    """
    将归一化的坐标转换为像素坐标。
    :param normalized_coords: 归一化的坐标列表 (x1, y1, x2, y2)
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 像素坐标列表 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = normalized_coords
    pixel_coords = [int(x1 * img_width), int(y1 * img_height),
                    int(x2 * img_width), int(y2 * img_height)]
    return pixel_coords

def process_yolo_txts(txt_dir, output_file, img_width, img_height):
    """
    处理YOLOv5的输出.txt文件，转换归一化的坐标为像素坐标，更改输出格式，并保存到新文件。
    :param txt_dir: 包含.txt文件的目录
    :param output_file: 输出文件路径
    :param img_width: 图片宽度
    :param img_height: 图片高度
    """
    with open(output_file, 'w') as out_f:
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_dir, filename)
                with open(file_path, 'r') as in_f:
                    for line in in_f:
                        data = line.strip().split()
                        class_index = int(data[0])
                        x1, y1, x2, y2 = map(float, data[1:5])
                        confidence = float(data[5])
                        pixel_coords = convert_normalized_to_pixel_coords([x1, y1, x2, y2], img_width, img_height)
                        class_name = class_names[class_index]
                        out_f.write(f"{pixel_coords[0]},{pixel_coords[1]},{pixel_coords[2]},{pixel_coords[3]},{confidence},{class_name}\n")

# 示例调用
txt_dir = 'path/to/your/txt/directory'
output_file = 'path/to/output.txt'
img_width = 640  # 图片宽度
img_height = 480  # 图片高度

# 假设类别名称如下：
class_names = ['business_license', 'tobacco_license', 'warning_board']

process_yolo_txts(txt_dir, output_file, img_width, img_height)