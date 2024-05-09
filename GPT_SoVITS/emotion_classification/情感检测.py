import os

# resp = request_AI(protmpts, "你好?")
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 语义分类
semantic_cls = None


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


replists = {
    "，", "。", "？", "！", "：", "—",
    ",", ".", "?", "!", "!", "~", ":", "—", "…",
    "|" , "、", "；", "：", "“", "”", "‘", "’",
    "《", "》",
    "（", "）",
    "【", "】",
    "〖", "〗",
    "［", "］",
    "｛", "｝",
    "〈", "〉",
    "﹏", "＿", "＠", "＃", "￥", "％",
    "\"", "\'", "‘", "’", "“", "”",
}


def get_semantic_cls(text):
    '''
    获取文本的情感分类
    :param text: 文本
    :return: 情感分类结果
    '''
    lt = text
    for rep in replists:
        lt = lt.replace(rep, " ")
    global semantic_cls
    if semantic_cls is None:
        EMOTION_MODEL_FILE = os.environ.get("EMOTION_MODEL_FILE")
        semantic_cls = pipeline(Tasks.text_classification,
                                EMOTION_MODEL_FILE, model_revision='v1.0.0')
    obj = semantic_cls(input=lt)
    # 重新排序
    ls = obj['labels']
    ss = obj['scores']
    obj['labels'], obj['scores'] = sort_labels_scores(ls, ss)
    return obj,lt


if __name__ == '__main__':
    input_labels = ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
    input_scores = [1, 2, 3, 4, 5, 6, 7]
    print(sort_labels_scores(input_labels, input_scores))
    input_labels = ['高兴', '喜好', '悲伤', '厌恶', '愤怒', '惊讶', '恐惧']
    input_scores = [2, 4, 3, 5, 6, 7, 1]
    print(sort_labels_scores(input_labels, input_scores))
    while True:
        text = input("请输入文本：")
        sl = get_semantic_cls(text)
        ps = sl['scores']
        pl = sl['labels']
        pr = {}
        for i in range(len(pl)):
            pr[pl[i]] = ps[i]
        print(pr)
