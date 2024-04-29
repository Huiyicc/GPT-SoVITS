import json
import os
import joblib
from 情感检测 import get_semantic_cls
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# 数据集路径
# data_file = r'F:\Engcode\AIAssistant\dataset\nahida\emotion.json'
emotion_data_file = os.environ.get("EMOTION_DATA_FILE")
with open(emotion_data_file, 'r', encoding='utf-8') as f:
    data_obj = json.loads(f.read())
# emotion模型路径
# model_file = r"F:\Engcode\AIAssistant\dataset\nahida\emotion_similarity_model.joblib"
model_file = os.environ.get("EMOTION_MODEL_FILE")

nparrs = []
knn = None

for obj in data_obj:
    emotion_arr = obj['emotion']["scores"]
    nparrs.append(emotion_arr)

# 检查model_file是否存在
if os.path.exists(model_file):
    knn = joblib.load(model_file)
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


def find_similar_sentences(text, K=1):
    # 假设有一个新的输入句子，其情绪分类输出scores向量为：
    raw_scores = get_semantic_cls(text)
    input_scores = np.array(raw_scores['scores'])

    # 对输入scores向量进行归一化处理
    normalized_input_scores = normalize(input_scores.reshape(1, -1))

    # 使用KNN模型查找与输入句子最相似的K个源数据
    distances, indices = knn.kneighbors(normalized_input_scores, n_neighbors=K)

    # 假设有原始数据集 `sentences`，其中句子按与scores向量对应的顺序排列
    closest_sentences = [data_obj[index] for index in indices[0]]
    return raw_scores,closest_sentences


if __name__ == '__main__':
    print(find_similar_sentences("谢谢你们，我很高兴。"))
