#coding=utf-8
import json
import os


def convert_labelme_to_yolo(json_file, output_txt_file, label_to_class_id):
    with open(json_file, 'r') as file:
        data = json.load(file)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    with open(output_txt_file, 'w') as file:
        for shape in data['shapes']:
            points = shape['points']
            class_id = label_to_class_id.get(shape['label'], -1)
            normalized_points = []
            for x, y in points:
                nx = round(x / image_width, 6)
                ny = round(y / image_height, 6)
                normalized_points.append(f"{nx} {ny}")
            file.write(f"{class_id} " + " ".join(normalized_points) + "\n")


def batch_convert(json_folder, output_folder, label_to_class_id):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_file = os.path.join(json_folder, filename)
            output_txt_file = os.path.join(output_folder, filename.replace('.json', '.txt'))
            convert_labelme_to_yolo(json_file, output_txt_file, label_to_class_id)


def create_label_file(label_dict, output_path):
    sorted_labels = [label for label, _ in sorted(label_dict.items(), key=lambda item: item[1])]

    with open(output_path, 'w') as file:
        for label in sorted_labels:
            file.write(f"{label}\n")


if __name__ == '__main__':
    # 使用示例
    label_to_class_id = {
        'road': 0,
        'car': 1,
        'tree': 2
    }

    json_folder = './datasets/before'
    output_folder = './datasets/before'

    batch_convert(json_folder, output_folder, label_to_class_id)
    # 生成lableme.txt文件，按照label_dict的值从小到大排列，一个类别1行
    create_label_file(label_to_class_id, './datasets/before/yolo-label.txt')
