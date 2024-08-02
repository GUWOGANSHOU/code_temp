import cv2
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def write_bbox(name, defect_name, bbox):
    img_path = './datasets/images/train/' + name
    if not os.path.exists(img_path):
        return img_path
    save_path = './show_bbox/' + defect_name + '_' + name
    img = cv2.imread(img_path)
    x, y, w, h =  bbox[0], bbox[1], bbox[2], bbox[3]
    x, y, w, h = int(x), int(y), int(w), int(h)
    x2, y2 = x + w, y + h
    img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
    img = cv2ImgAddText(img, defect_name, x, y, (255, 0, 0), 40)
    # cv2.imwrite(save_path, img)
    # cv2.imencode(保存格式, 保存图片)[1].tofile(保存中文路径)
    cv2.imencode('.jpg', img)[1].tofile(save_path)


josn_path = "./datasets/anno_train.json"
image_path = "./datasets/images/train/"
with open(josn_path, 'r') as f:
    temps = tqdm(json.loads(f.read()))
    for temp in temps:
        name= temp['name']
        defect_name = temp['defect_name']
        bbox = temp['bbox']
        write_bbox(name, defect_name, bbox)







# im_path = './datasets/images/train/0a5ddaea807e6d8d1219350183.jpg'
# img = cv2.imread(im_path)
# print(img)
# defect_name = "\u6d46\u6591"
# defect_name.encode().decode('unicode_escape')
# print(defect_name)
#
# x, y, w, h =  140.48, 41.6, 489.33, 240.58
# x, y, w, h = int(x), int(y), int(w), int(h)
# x2, y2 = x + w, y + h
# img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
# img = cv2ImgAddText(img, defect_name, x, y, (255, 0, 0), 40)
# cv2.imshow('defect_show', img)
# cv2.imwrite('./show_bbox/0a5ddaea807e6d8d1219350183.jpg', img)
# key = cv2.waitKey(0)



