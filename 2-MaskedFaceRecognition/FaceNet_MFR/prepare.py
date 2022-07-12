import os
import shutil
import random


def generate_val_data(lib_dir, val_dir):
    # 写入文件
    f = open('data/val_gt.json', 'w')
    f.write('{')
    for category in os.listdir(lib_dir):
        source_dir = os.path.join(lib_dir, category)
        if not os.path.isdir(source_dir):
            os.mkdir(source_dir)
        for image in os.listdir(source_dir):
            n = random.random()
            if n < 0.1:
                shutil.copy(source_dir + '/' + image, val_dir + '/' + image)
                f.write('"%s":"%s",' % (image, category))
    f.write('}')
    f.close()


if __name__ == '__main__':
    print('Start')
    # generate_val_data('data/lib', 'data/val')
