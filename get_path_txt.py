import os

def gci(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        else:
            print(fi_d)
            # str = os.path.join(filepath, fi_d)[:-4] + '.jpg' + '\n'
            str = os.path.join(filepath, fi_d) + '\n'
            list_txt.writelines(str)

# if not os.path.exists('datasets/labels/'):
#     os.makedirs('datasets/labels/')
#
# list_txt = open('datasets/labels/train.txt', 'w')
# path = r'D:\yqh\jupyter_notebook\yolov7-main\datasets\images\train'
if not os.path.exists('datasets/custom_dispenser/labels/'):
    os.makedirs('datasets/custom_dispenser/labels/')
list_txt = open('datasets/custom_dispenser/labels/val_custom_dispenser.txt', 'w')
path = r'D:\yqh\jupyter_notebook\yolov7-main\datasets\custom_dispenser\images\val'
# list_txt = open('datasets/mini-coco2017/val.txt', 'w')
# path = r'D:\yqh\jupyter_notebook\yolov7-main\datasets\mini-coco2017\val2017'
gci(path)
list_txt.close()
