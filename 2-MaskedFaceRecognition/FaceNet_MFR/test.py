import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray, expand_dims
from keras.models import load_model
import matplotlib.pyplot as plt


def get_embedding(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(required_size)
    pixels = asarray(image).astype('float32')
    # 标准化
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    # 扩展维数，将数组转化为样本
    samples = expand_dims(pixels, axis=0)
    # 预测以获取向量
    embedding = model.predict(samples)[0]
    return embedding


def calculate_distance(embedding, pair_embedding):
    distance = 0
    if paradigm == 'l1':
        distance = np.sum(np.abs(embedding - pair_embedding))
    elif paradigm == 'l2':
        distance = np.sqrt(np.sum(np.square(embedding - pair_embedding)))
    return distance


def get_distances(path):
    df = pd.read_csv(path, header=0)
    distances = list()
    for i in range(min(5000, len(df))):
        path = df['path'][i]
        pair_path = df['pair_path'][i]
        distance = calculate_distance(get_embedding(path), get_embedding(pair_path))
        distances.append(distance)
        if i % 200 == 0:
            print(i)
    return np.asarray(distances)


def get_threshold(same_path, diff_path):
    same = get_distances(same_path)
    diff = get_distances(diff_path)
    threshold = 0
    if standard == 'mode':
        same_hist = plt.hist(same, 100, range=[np.floor(np.min([same.min(), diff.min()])),
                                               np.ceil(np.max([same.max(), diff.max()]))], alpha=0.5, label='same')
        diff_hist = plt.hist(diff, 100, range=[np.floor(np.min([same.min(), diff.min()])),
                                               np.ceil(np.max([same.max(), diff.max()]))], alpha=0.5, label='diff')
        difference = same_hist[0] - diff_hist[0]
        difference[:same_hist[0].argmax()] = np.Inf
        difference[diff_hist[0].argmax():] = np.Inf
        threshold = (same_hist[1][np.where(difference <= 0)[0].min()] +
                     same_hist[1][np.where(difference <= 0)[0].min() - 1]) / 2
    elif standard == 'average':
        threshold = (np.average(same) + np.average(diff)) / 2
    print('threshold is', threshold)
    return threshold


def evaluate(test_path, threshold):
    df = pd.read_csv(test_path, header=0)
    match_count = 0
    for i in range(len(df)):
        target = df['target'][i]
        path = df['path'][i]
        pair_target = df['pair_target'][i]
        pair_path = df['pair_path'][i]
        distance = calculate_distance(get_embedding(path), get_embedding(pair_path))
        if (target == pair_target and distance <= threshold) or (target != pair_target and distance > threshold):
            match_count = match_count + 1
    accuracy = match_count / len(df)
    print('accuracy is', accuracy)
    return accuracy


def assess(assess_path, threshold):
    df = pd.read_csv(assess_path, header=0)
    res = {}
    for i in range(len(df)):
        path = df['path'][i]
        pair_path = df['pair_path'][i]
        distance = calculate_distance(get_embedding(path), get_embedding(pair_path))
        print("\r%.2f%%" % (i / len(df) * 100), end="")
        if distance < threshold:
            res[str(i + 1)] = 1
        else:
            res[str(i + 1)] = 0
    with open('result/assess.json', 'w') as f_json:
        f_json.write(json.dumps(res))
        f_json.close()
    with open('result/assess.txt', 'w') as f_txt:
        for i in range(len(res)):
            f_txt.write(str(i + 1) + " " + str(res[str(i + 1)]) + "\n")
        f_txt.close()


if __name__ == '__main__':
    model = load_model('model/facenet_inception_resnet_v1.h5')
    paradigm = 'l1'
    standard = 'mode'
    # l1 mode 101.6999 0.9224
    # l1 average 97.95 0.9178
    # l2 mode 11.1     0.9222
    # l2 average 10.81 0.9172
    # evaluate('data/test.csv', get_threshold('data/eval_same.csv', 'data/eval_diff.csv'))
    assess('data/assess.csv', 101.7)
