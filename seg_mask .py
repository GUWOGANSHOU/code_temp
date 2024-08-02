# labelme安装位置（D:\yqh\software\anaconda3\Lib\site-packages\labelme\cli）的json_to_dataset.py文件，修改为：
import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')  # 标注文件json所在的文件夹
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)  # 获取json文件列表
    out_dir = 'mask'
    out_dir = osp.join(osp.dirname(json_file), out_dir)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])  # 获取每个json文件的绝对路径
        filename = list[i][:-5]  # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        extension = list[i][-4:]
        if extension == 'json':
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
                # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data[
                    'shapes'])  # data['shapes']是json文件中记录着标注的位置及label等信息的字段

                # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                # lbl_viz = utils.draw.draw_label(lbl, img, captions)
                '''
                out_dir = osp.basename(list[i])[:-5] + '_json'
                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)
                '''

                # PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_source.png'.format(filename)))
                PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}.png'.format(filename)))
                # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))
                '''
                with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                    for lbl_name in lbl_names:
                        f.write(lbl_name + '\n')

                warnings.warn('info.yaml is being replaced by label_names.txt')
                info = dict(label_names=lbl_names)
                with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                    yaml.safe_dump(info, f, default_flow_style=False)
                '''

                print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()


# 修改后在conda中输入
# cd D:\yqh\software\anaconda3\Scripts
# labelme_json_to_dataset.exe D:/test_json

# 转换彩图
import os
from PIL import Image

filepath=r'D:\mask'
total = os.listdir(filepath)
savepath = r'D:\test_json'

num = len(total)
list = range(num)


for i in list:
    filename = total[i][:-4] + '.png'
    img = os.path.join(filepath, filename)
    mask = Image.open(img).convert('L')
    mask.putpalette([0, 0, 0,  # putpalette给对象加上调色板，相当于上色：背景为黑色，目标１为红色，目标2为黄色，目标3为橙色（如果你的图中有更多的目标，可以自行添加更多的调色值）
                     255, 255, 255,
                     255, 255, 0,
                     255, 153, 0])

    #filename2 = total[i][:-4] + '.png'
    mask.save(os.path.join(savepath, filename))

