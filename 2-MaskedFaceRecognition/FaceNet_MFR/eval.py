import json

# 预测生成的json文件地址
# pred_json_path = "result/val_TripletLoss.json"
pred_json_path = "result/val_ArcFace.json"
# GT标注的json文件地址
gt_json_path = "data/val_gt.json"
# 用于记录人脸识别正确的图像数
correct = 0

# 读取json文件内容,返回字典格式
with open(pred_json_path, 'r') as f1, open(gt_json_path, 'r') as f2:
    pred_dic = json.load(f1)
    gt_dic = json.load(f2)

    # 记录总图像数
    all = len(gt_dic)

    for key in gt_dic.keys():
        # 分别获取真实标注和预测值
        gt_value = gt_dic[key]
        pred_value = pred_dic[key]

        # 两个结果相同则识别正确
        if pred_value == gt_value:
            correct = correct + 1

        continue

    # 计算准确率指标
    accuracy = round(correct / all, 4)
    print(accuracy)

    # 关闭文件读写
    f1.close()
    f2.close()





