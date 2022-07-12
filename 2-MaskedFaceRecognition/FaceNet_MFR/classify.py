import numpy as np


def predict(library_path, target_path, res_path):
    # 获取人脸特征库
    library = np.load(library_path, allow_pickle=True)
    library_feature, library_label = library['arr_0'], library['arr_1']
    print('Feature Library Loaded:', library_feature.shape)
    # 获取待预测图片特征
    target = np.load(target_path, allow_pickle=True)
    target_feature, target_label = target['arr_0'], target['arr_1']
    print('Predict Target Loaded:', target_feature.shape)
    # 写入文件
    f = open(res_path, 'w')
    f.write('{')
    # 预测身份
    for i in range(len(target_feature)):
        min_distance = 100
        predict_label = ''
        for j in range(len(library_feature)):
            distance = np.sqrt(np.sum(np.square(target_feature[i] - library_feature[j])))
            if distance < min_distance:
                min_distance = distance
                predict_label = library_label[j]
        print('%s %s %.3f' % (target_label[i], predict_label, min_distance))
        f.write('"%s":"%s"' % (target_label[i], predict_label))
        if i < len(target_feature) - 1:
            f.write(',')
        else:
            f.write('}')
    f.close()


if __name__ == '__main__':
    print('Start')
    # predict('result/lib_TripletLoss.npz', 'result/val_TripletLoss.npz', 'result/val_TripletLoss.json')
    # predict('result/lib_TripletLoss.npz', 'result/test_TripletLoss.npz', 'result/test_TripletLoss.json')
    # predict('result/lib_ArcFace.npz', 'result/val_ArcFace.npz', 'result/val_ArcFace.json')
    predict('result/lib_ArcFace.npz', 'result/test_ArcFace.npz', 'result/test_ArcFace.json')
    # predict('result/lib_RetrainedArcFace.npz', 'result/val_RetrainedArcFace.npz', 'result/val_RetrainedArcFace.json')

