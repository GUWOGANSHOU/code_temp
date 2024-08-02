import stat

import albumentations as A
import cv2
import os

"""
该脚本主要实现了利用albumentations工具包对yolo标注数据进行增强
给定一个存放图像和标注文件的主目录，在主目录下自动生成增强的图像和标注文件
"""



def get_enhance_save(old_images_files, old_labels_files, label_list, enhance_images_files, enhance_labels_files):
    # 这里设置指定的数据增强方法
    transform = A.Compose([
        # 随机更改图像的颜色，饱和度
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1.0),
        # # 随机仿射变换
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.5),
        A.GaussNoise(p=0.1),  # 将高斯噪声应用于输入图像。

        # # A.Resize(width=1223, height=500),  # 调整图像大小
        # #　裁剪图像的随机部分，确保裁剪的部分将包含原始图像的所有边界框，erosion_rate参数控制裁剪后可能丢失原始边界框的面积
        # A.RandomSizedBBoxSafeCrop(width=480, height=240, erosion_rate=0.1),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2), # 随机亮度和对比度调整

        # A.VerticalFlip(p=0.5), # 垂直翻转
        # A.HorizontalFlip(p=0.5),  # 水平翻转
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.60, rotate_limit=60, p=0.75), # 随机仿射变换
        # A.OneOf([
        #     A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
        #     A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
        #     A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        # ], p=0.2),
        # A.ChannelShuffle(always_apply=False, p=0.5),
        # A.ToGray(p=1.0),
        A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.5),  # 随机亮度和对比度调整
            ], p=1.0),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=1.0),
        A.RandomSizedBBoxSafeCrop(width=500, height=160, erosion_rate=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.3, label_fields=['class_labels']))

    # 这里指定修改后image和label的文件名
    mid_name = "ToGray_Flip_ShiftScaleRotate_GaussNoise"
    # mid_name = "RandomSizedBBoxSafeCrop_Flip_RandomBrightnessContrast"
    # mid_name = "ShiftScaleRotate_Flip_Blur"

    label_files_name = os.listdir(old_labels_files)

    for name in label_files_name:

        label_files = os.path.join(old_labels_files, name)

        # os.chmod(label_files, stat.S_IRWXU)

        yolo_b_boxes = open(label_files).read().splitlines()

        bboxes = []

        class_labels = []

        # 对一个txt文件的每一行标注数据进行处理
        for b_box in yolo_b_boxes:
            b_box = b_box.split(" ")
            m_box = b_box[1:5]

            m_box = list(map(float, m_box))

            m_class = b_box[0]

            bboxes.append(m_box)
            class_labels.append(label_list[int(m_class)])

        # 读取对应的图像
        image_path = os.path.join(old_images_files, name.replace(".txt", ".jpg"))
        if os.path.exists(image_path) is False:
            image_path = os.path.join(old_images_files, name.replace(".txt", ".jpg"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调用上面定义的图像增强方法进行数据增强
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        transformed_b_boxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # 先判断目标文件夹路径是否存在
        if os.path.exists(enhance_images_files) is False:
            os.mkdir(enhance_images_files)
        a, b = os.path.splitext(name)
        new_name = a + mid_name + b
        cv2.imwrite(os.path.join(enhance_images_files, new_name.replace(".txt", ".jpg")), transformed_image)

        if os.path.exists(enhance_labels_files) is False:
            os.mkdir(enhance_labels_files)

        new_txt_file = open(os.path.join(enhance_labels_files, new_name), "w")

        new_bboxes = []

        for box, label in zip(transformed_b_boxes, transformed_class_labels):

            new_class_num = label_list.index(label)
            box = list(box)
            for i in range(len(box)):
                box[i] = str(('%.5f' % box[i]))
            box.insert(0, str(new_class_num))
            new_bboxes.append(box)

        for new_box in new_bboxes:

            for ele in new_box:
                if ele is not new_box[-1]:
                    new_txt_file.write(ele + " ")
                else:
                    new_txt_file.write(ele)

            new_txt_file.write('\n')

        new_txt_file.close()


def main():
    root = r"D:\yqh\jupyter_notebook\yolov7-main\datasets"
    old_images_files = os.path.join(root, "kitchen\\images\\train")
    old_labels_files = os.path.join(root, "kitchen\\labels\\train")

    enhance_images_files = os.path.join(root, "kitchen\\enhance_images")
    enhance_labels_files = os.path.join(root, "kitchen\\enhance_labels")

    # 这里设置数据集的类别
    # label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

    label_list = ['smoking', 'shirtless', 'mouse', 'cat', 'dog']
    # 实现对传入的数据文件进行遍历读取，并进行数据增强
    get_enhance_save(old_images_files, old_labels_files, label_list, enhance_images_files, enhance_labels_files)


if __name__ == '__main__':
    main()