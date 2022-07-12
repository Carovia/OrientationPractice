import glob
import math
import os
import random
import xml.etree.ElementTree as ET


def random_split(total_path, train_path, val_path):
    total_txt = open(total_path, 'r')
    train_txt = open(train_path, 'w')
    val_txt = open(val_path, 'w')
    total = total_txt.readlines()
    random.shuffle(total)
    train_sum = math.floor(len(total) * 0.8)
    for i in range(len(total)):
        if i < train_sum:
            train_txt.writelines(total[i])
        else:
            val_txt.writelines(total[i])
    total_txt.close()
    train_txt.close()
    val_txt.close()


def xml_to_txt(xml_dir, txt_path, image_dir):
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    f = open(txt_path, 'w')
    for xml_file in xml_files:
        image_path = image_dir + '/' + os.path.basename(xml_file)[:-4] + ET.parse(xml_file).find('filename').text[-4:]
        objects = ET.parse(xml_file).findall('object')
        f.write("%s " % image_path)
        for obj in objects:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            if obj.find('name').text == 'mask':
                category = 0
            else:
                category = 1
            f.write("%d,%d,%d,%d,%d " % (x1, y1, x2, y2, category))
        f.write("\n")
    f.close()


xml_to_txt('datasets/train/Annotations', 'model_data/total.txt', 'datasets/train/JPEGImages')
random_split('model_data/total.txt', 'model_data/train.txt', 'model_data/val.txt')