import os
import json

import librosa
os.environ["EMOTION_MODEL_FILE"] = r'GPT_SoVITS/emotion_classification/nlp_structbert_emotion-classification_chinese-large'
from 情感检测 import get_semantic_cls

# 源目录
# lab文件目录
lab_path = r''
# wav文件目录
wav_path = r''
# 输出目录
out_path = r'F:\Engcode\AIAssistant\dataset\nahida'



out_obj = []


def test_read_all_byss():
    # 获取所有文件
    files = [f for f in os.listdir(lab_path) if os.path.isfile(os.path.join(lab_path, f)) and f.endswith('.lab')]
    i = 0
    max = len(files)
    for file in files:
        with open(os.path.join(lab_path, file), 'r', encoding='utf-8') as f:
            content = f.read()
        s = get_semantic_cls(content)
        # 删除后缀
        file_name = file.split('.')[0]
        ldata = {
            'file': file_name,
            'content': content,
            'emotion': s
        }
        # 跟随gsv规则过滤不符合时长的音频
        wav16k, sr = librosa.load(os.path.join(wav_path, file_name+".wav"), sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            i += 1
            print(f'continue [{i / max * 100}%]{i}/{max}')
            continue
        out_obj.append(ldata)
        i += 1
        print(f'append [{i / max * 100}%]{i}/{max}')


test_read_all_byss()

with open(out_path + '/emotion.json', 'w', encoding='utf-8') as f_out:
    f_out.write(json.dumps(out_obj, ensure_ascii=False, indent=4))
