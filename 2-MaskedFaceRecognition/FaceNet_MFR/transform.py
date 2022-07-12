import os
import numpy
from PIL import Image
from numpy import asarray, expand_dims, savez_compressed
from keras.models import load_model
import torch
from model import FaceNet2


# 从给定的图像中提取人脸
def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# 对图像获取人脸嵌入
def get_embedding(pixels, model, model_name):
    pixels = pixels.astype('float32')
    # 标准化
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    # 扩展维数，将数组转化为样本
    samples = expand_dims(pixels, axis=0)
    # 预测以获取向量
    embedding = None
    if model_name == 'TripletLoss':
        embedding = model.predict(samples)[0]
    if model_name == 'ArcFace' or model_name == 'RetrainedArcFace':
        embedding = model(torch.tensor(samples.transpose((0, 3, 1, 2))))['embeddings']
        embedding = embedding.detach().numpy()[0]
    return embedding


# 加载数据集
def load_dataset(data_path, data_type, model, model_name):
    embeddings, labels = list(), list()
    if data_type == 'lib':
        for category in os.listdir(data_path):
            category_path = os.path.join(data_path, category)
            category_embeddings = None
            count = 0
            for image_name in os.listdir(category_path):
                face = extract_face(os.path.join(category_path, image_name))
                if count == 0:
                    category_embeddings = get_embedding(face, model, model_name)
                else:
                    category_embeddings = category_embeddings + get_embedding(face, model, model_name)
                count = count + 1
            if count == 0:
                print('No Faces of %s' % category)
                continue
            embeddings.append(category_embeddings / count)
            labels.append(category)
            print('Features from %d Images Averaged for %s' % (count, category))
    else:
        for image_name in os.listdir(data_path):
            face = extract_face(os.path.join(data_path, image_name))
            embeddings.append(get_embedding(face, model, model_name))
            labels.append(image_name)
            print("Features Extracted from Image %s" % image_name)
    embeddings = asarray(embeddings)
    labels = asarray(labels)
    savez_compressed('result/%s_%s.npz' % (data_type, model_type), embeddings, labels)
    print("Dataset Transformed:", embeddings.shape, labels.shape)


def choose_model(model_name):
    if model_name == 'TripletLoss':
        print('Model Loaded')
        return load_model('model/facenet_inception_resnet_v1.h5')
    if model_name == 'ArcFace':
        print('Model Loaded')
        return torch.load('model/InceptionResNetV1_ArcFace.pt', map_location=torch.device('cpu'))
    if model_name == 'RetrainedArcFace':
        print('Model Loaded')
        return torch.load('model/Retrained_InceptionResNetV1_ArcFace.pt', map_location=torch.device('cpu'))['model']
    else:
        print('No Available Model')
        return None


if __name__ == '__main__':
    model_type = 'TripletLoss'
    # model_type = 'ArcFace'
    my_model = choose_model(model_type)
    print(my_model)
    # load_dataset('data/lib', 'lib', my_model, model_type)
    # load_dataset('data/val', 'val', my_model, model_type)
    # load_dataset('data/test', 'test', my_model, model_type)
