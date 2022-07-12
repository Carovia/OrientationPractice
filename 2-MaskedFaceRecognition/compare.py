import time
import json
import pandas as pd
from PIL import Image


def compare(file_path_1, file_path_2):
    f1 = open(file_path_1, 'r')
    f2 = open(file_path_2, 'r')
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    count = 0
    ids = []
    for i in range(len(lines1)):
        if lines1[i] != lines2[i]:
            print(i+1)
            ids.append(i)
            count = count + 1
    print()
    print(count)
    print(len(lines1))


def vote(file_path_1, file_path_2, file_path_3, res_path):
    f1 = open(file_path_1, 'r')
    f2 = open(file_path_2, 'r')
    f3 = open(file_path_3, 'r')
    l1 = f1.readlines()
    l2 = f2.readlines()
    l3 = f3.readlines()
    res = {}
    ids = []
    diff_count = 0
    for i in range(len(l1)):
        if l1[i] == l2[i] == l3[i]:
            print(i + 1, 'same')
            res[str(i + 1)] = int(l1[i].split(' ')[1])
        else:
            ids.append(i)
            diff_count = diff_count + 1
            score = int(l1[i].split(' ')[1]) + int(l2[i].split(' ')[1]) + int(l3[i].split(' ')[1])
            if score >= 2:
                print(i + 1, 'diff 1')
                res[str(i + 1)] = 1
            else:
                print(i + 1, 'diff 0')
                res[str(i + 1)] = 0
    print()
    print(diff_count)
    print(len(l1))
    with open(res_path, 'w') as f_json:
        f_json.write(json.dumps(res))
        f_json.close()
    return res, ids


def check(assess_csv, res, ids):
    df = pd.read_csv(assess_csv, header=0)
    for idx in ids:
        path = './FaceNet_MFR' + df['path'][idx][1:]
        pair_path = './FaceNet_MFR' + df['pair_path'][idx][1:]
        Image.open(path).show()
        Image.open(pair_path).show()
        print(res[str(idx + 1)])
        time.sleep(2)


if __name__ == '__main__':
    compare('Trainable_MFR/result/assess.txt', 'FaceNet_MFR/result/assess.txt')
    # compare('Trainable_MFR/result/assess.txt', 'Trainable_MFR/result/assess-retrained.txt')
    # compare('Trainable_MFR/result/assess-retrained.txt', 'FaceNet_MFR/result/assess.txt')
    vote_res, vote_ids = vote('Trainable_MFR/result/assess.txt', 'Trainable_MFR/result/assess-retrained.txt',
                              'FaceNet_MFR/result/assess.txt', 'assess-final.json')
    # check('FaceNet_MFR/data/assess.csv', vote_res, vote_ids)
