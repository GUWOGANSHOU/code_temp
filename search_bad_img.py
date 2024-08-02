
import imghdr
import os


def main(root):
    path_list = os.listdir(root)
    items = {"bed": 0, "bed_list": []}

    def is_verify(path):
        if imghdr.what(path):
            print('Good img',path)
        else:
            items["bed"] += 1
            items["bed_list"].append(path)
            os.remove(path)

    for path in path_list:
        is_verify(root + '/' + path)
    print(f'总计删除损坏文件{items["bed"]}个.')
    print("损坏文件如下:", "\n".join(items['bed_list']))


if __name__ == '__main__':
    main('./datasets/dw\images/val')

