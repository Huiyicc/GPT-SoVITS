# import os
# from os import listdir
# from os.path import isfile, join
#
# from api_gen import request_AI
#
# system_prompt = '''你是一个乐于助人的助手.
# '''
#
# ucontent = '''你好？
# '''
#
# protmpts = {
#     "temperature": 0.5,
#     "frequency_penalty": 0,
#     "presence_penalty": 0,
#     "system_prompt": system_prompt,
#     "injecting_input": False,
#     "message": [
#         {
#             "role": "user",
#             "content": "language tutor is you?",
#         },
#         {
#             "assistants": True,
#             "contents": [
#             ]
#         },
#         {
#             "role": "user",
#             # "prompt": "prompts/as_ai_dict_sentence_word.prompt"
#             "content": ucontent,
#         }
#     ]
# }

# resp = request_AI(protmpts, "你好?")
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 语义分类
semantic_cls = pipeline(Tasks.text_classification,
                        r'G:\models\bert\nlp_structbert_emotion-classification_chinese-large', model_revision='v1.0.0')


def sort_labels_scores(labels, scores):
    '''
    对情感分类结果进行排序
    :param labels: 情感分类标签
    :param scores: 情感分类得分
    :return: 排序后的标签和得分
    '''
    label_score_dict = dict(zip(labels, scores))
    fixed_order = ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
    sorted_pairs = sorted(label_score_dict.items(), key=lambda x: fixed_order.index(x[0]))
    sorted_labels, sorted_scores = zip(*sorted_pairs)
    return list(sorted_labels), list(sorted_scores)


def get_semantic_cls(text):
    '''
    获取文本的情感分类
    :param text: 文本
    :return: 情感分类结果
    '''
    obj = semantic_cls(input=text)
    # 重新排序
    ls = obj['labels']
    ss = obj['scores']
    obj['labels'], obj['scores'] = sort_labels_scores(ls, ss)
    return obj


if __name__ == '__main__':
    input_labels = ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
    input_scores = [1, 2, 3, 4, 5, 6, 7]
    print(sort_labels_scores(input_labels, input_scores))
    input_labels = ['高兴', '喜好', '悲伤', '厌恶', '愤怒', '惊讶', '恐惧']
    input_scores = [2, 4, 3, 5, 6, 7, 1]
    print(sort_labels_scores(input_labels, input_scores))
