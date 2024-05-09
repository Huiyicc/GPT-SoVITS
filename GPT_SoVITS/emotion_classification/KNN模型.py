import json
import os
import joblib
from GPT_SoVITS.emotion_classification.情感检测 import get_semantic_cls
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# 数据集路径
# data_file = r'F:\Engcode\AIAssistant\dataset\nahida\emotion.json'
emotion_data_file = os.environ.get("EMOTION_DATA_FILE")
with open(emotion_data_file, 'r', encoding='utf-8') as f:
    data_obj = json.loads(f.read())
# emotion模型路径
# model_file = r"F:\Engcode\AIAssistant\dataset\nahida\emotion_similarity_model"
model_file_doubt = os.environ.get("EMOTION_JOBLIB_FILE") + "_doubt.joblib"
model_file_normal = os.environ.get("EMOTION_JOBLIB_FILE") + "_normal.joblib"

nparrs_doubt = []
nparrs_normal = []

knn_doubt = None
knn_normal = None

for obj in data_obj["doubt"]:
    emotion_arr = obj['emotion']["scores"]
    nparrs_doubt.append(emotion_arr)

for obj in data_obj["normal"]:
    emotion_arr = obj['emotion']["scores"]
    nparrs_normal.append(emotion_arr)


def init_model(model_file, nparrs):
    # 检查model_file是否存在
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        train_scores = np.array(nparrs)
        # 对scores向量进行归一化
        normalized_train_scores = normalize(train_scores, axis=1)
        # 构建KNN模型，使用余弦相似度作为距离度量
        knn = NearestNeighbors(metric='cosine', algorithm='brute')

        # 使用训练数据拟合模型
        knn.fit(normalized_train_scores)
        # 保存模型
        joblib.dump(knn, model_file)
        return knn


knn_doubt = init_model(model_file_doubt, nparrs_doubt)
knn_normal = init_model(model_file_normal, nparrs_normal)


def find_similar_sentences(text, K=5):
    # 假设有一个新的输入句子，其情绪分类输出scores向量为：
    raw_scores, _ = get_semantic_cls(text)
    input_scores = np.array(raw_scores['scores'])

    # 对输入scores向量进行归一化处理
    normalized_input_scores = normalize(input_scores[np.newaxis, :])

    # 使用KNN模型查找与输入句子最相似的K个源数据
    if text.find('？') != -1 or text.find('?') != -1:
        distances, indices = knn_doubt.kneighbors(normalized_input_scores, n_neighbors=K)
        c_key = "doubt"
    else:
        distances, indices = knn_normal.kneighbors(normalized_input_scores, n_neighbors=K)
        c_key = "normal"

    # 假设有原始数据集 `sentences`，其中句子按与scores向量对应的顺序排列
    closest_sentences = [data_obj[c_key][index] for index in indices[0]]
    return raw_scores, closest_sentences


if __name__ == '__main__':
    while True:
        text = input("请输入文本：")
        print(find_similar_sentences(text))
        print()
