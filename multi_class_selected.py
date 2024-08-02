# coding=utf-8
import os
import re
import shutil

# 给定文件夹，返回存在指定类别列表中任一类别的文件名
def get_class_exist_path(l_path, classes):
    result = []
    for root, dirs, files in os.walk(l_path):
        total = len(files)
        for i, file in enumerate(files):
            f_path = os.path.join(root, file)
            with open(f_path) as f:
                for line in f.readlines():
                    items = re.split(r"[ ]+", line.strip())
                    if items[0] in classes:
                        result.append(file)
                        break
            print("检查类别进度: %d/%d" % (i + 1, total))
    return result

def move_txt_files(l_path, e_fs, dst_d_path):
    total = len(e_fs)
    for i, e_f in enumerate(e_fs):
        src = os.path.join(l_path, e_f)
        dst = os.path.join(dst_d_path, e_f)
        shutil.move(src, dst)
        print("移动文本文件: %d/%d" % (i+1, total))

def delete_other_class(dist_l_path, classes):
    for root, dirs, files in os.walk(dist_l_path):
        total = len(files)
        for i, file in enumerate(files):
            f_path = os.path.join(dist_l_path, file)
            with open(f_path, 'r') as f:
                lines = f.readlines()
            with open(f_path, 'w') as f_w:
                for line in lines:
                    items = re.split(r"[ ]+", line.strip())
                    if items[0] in classes:
                        f_w.write(line)
            print("删除其他类别进度: %d/%d" % (i + 1, total))

def move_images(img_dir, dist_img_dir, dist_l_path):
    for root, dirs, files in os.walk(dist_l_path):
        total = len(files)
        for i, file in enumerate(files):
            image_file = os.path.splitext(file)[0] + ".jpg"
            src = os.path.join(img_dir, image_file)
            dst = os.path.join(dist_img_dir, image_file)
            shutil.move(src, dst)
            print("移动图片文件: %d/%d" % (i+1, total))

# 用于类别重映射的函数
def remap_categories(dist_l_path, old_classes, new_classes):
    category_map = dict(zip(old_classes, new_classes))
    for root, dirs, files in os.walk(dist_l_path):
        total = len(files)
        for i, file in enumerate(files):
            f_path = os.path.join(dist_l_path, file)
            with open(f_path, 'r') as f:
                lines = f.readlines()
            with open(f_path, 'w') as f_w:
                for line in lines:
                    items = re.split(r"[ ]+", line.strip())
                    items[0] = str(category_map[items[0]])  # 将类别ID重映射
                    f_w.write(' '.join(items) + '\n')  # 写回更新后的行
            print("类别重映射进度: %d/%d" % (i + 1, total))

# 原始标签文件路径
labels_path = r"D:\yqh\jupyter_notebook\yolov7-main\datasets\coco2017\coco2017\labels\train2017"
# 目标标签文件路径
dist_labels_path = r"D:\yqh\jupyter_notebook\yolov7-main\datasets\coco2017\coco2017\labels\train"
# 目标类别列表，0: person 56: chair 57: couch 60: dining table 62: tv 63: laptop
# ['person', 'chair', 'couch', 'dining table', 'tv', 'laptop']
old_classes = ['0', '56', '57', '60', '62', '63']
new_classes = list(map(str, range(len(old_classes))))  # 生成新的连续类别ID

# 第一步：将含有指定class列表中任一class的txt文件移动到指定目录
e_files = get_class_exist_path(labels_path, old_classes)
move_txt_files(labels_path, e_files, dist_labels_path)

# 第二步：删除移动后的txt中，不属于指定class列表的class
delete_other_class(dist_labels_path, old_classes)

# 第三步：将含有指定class列表中任一class的images移动到指定目录
images_dir = r"D:\yqh\jupyter_notebook\yolov7-main\datasets\coco2017\coco2017\images\train2017"
dist_images_dir = r"D:\yqh\jupyter_notebook\yolov7-main\datasets\coco2017\coco2017\images\train"
move_images(images_dir, dist_images_dir, dist_labels_path)

# 第四步：类别重映射
remap_categories(dist_labels_path, old_classes, new_classes)