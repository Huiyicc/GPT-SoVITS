import os
import json
from 情感检测 import get_semantic_cls

# 源目录,lab文件目录
path = r''
# 输出目录
out_path = r'output'



out_obj = []


def test_read_all_byss():
    # 获取所有文件
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.lab')]
    i = 0
    max = len(files)
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            content = f.read()
        s = get_semantic_cls(content)
        # 删除后缀
        file_name = file.split('.')[0]
        ldata = {
            'file': file_name,
            'content': content,
            'emotion': s
        }

        out_obj.append(ldata)
        i += 1
        print(f'[{i / max * 100}%]{i}/{max}')


test_read_all_byss()

with open(out_path + '/emotion.json', 'w', encoding='utf-8') as f_out:
    f_out.write(json.dumps(out_obj, ensure_ascii=False, indent=4))
