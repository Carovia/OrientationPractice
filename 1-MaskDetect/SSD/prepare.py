import glob
import math
import os
import random
import xml.etree.ElementTree as ET


def create_txt(image_dir, txt_path):
    f = open(txt_path, 'w')
    for image in os.listdir(image_dir):
        f.writelines(image[:-4] + '\n')
    f.close()


def random_split(src_path, res1_path, res2_path):
    src = open(src_path, 'r')
    res1 = open(res1_path, 'w')
    res2 = open(res2_path, 'w')
    total = src.readlines()
    random.shuffle(total)
    sum1 = math.floor(len(total) * 0.8)
    for i in range(len(total)):
        if i < sum1:
            res1.writelines(total[i])
        else:
            res2.writelines(total[i])
    src.close()
    res1.close()
    res2.close()


def xml_to_txt(xml_dir, txt_dir):
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
        image_name = os.path.basename(xml_file)
        txt_path = os.path.join(txt_dir, image_name[:-4] + '.txt')
        f = open(txt_path, 'w')
        objects = ET.parse(xml_file).findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            f.write("%s %d %d %d %d\n" % (obj.find('name').text, x1, y1, x2, y2))
        f.close()


# create_txt('datasets/JPEGImages', 'datasets/ImageSets/Main/all.txt')
# random_split('datasets/VOC2007/ImageSets/Main/trainval.txt',
#              'datasets/VOC2007/ImageSets/Main/train.txt', 'datasets/VOC2007/ImageSets/Main/val.txt')

# xml_to_txt('SSD/datasets/val/Annotations', 'SSD/datasets/val/GroundTruth')