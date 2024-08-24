import os

source_path = '/root/autodl-fs/datasets/流萤'
info_list = []


def check_file(name, pos):
    if len(name) > len(pos):
        end = name[len(name) - len(pos):]
        return end == pos, end
    return False, ""


def init_list(path):
    for name in os.listdir(path):
        check, pos = check_file(name, ".lab")
        if not check:
            continue
        with open(os.path.join(source_path, name), 'r', encoding='utf-8') as f:
            s = f.read()
            t_obj = {
                'text': s,
                'name': os.path.basename(name).split('.')[0],
            }
            info_list.append(t_obj)


init_list(source_path)
print(info_list)

for obj in info_list:
    t: str = obj['text']
    index = t.find('那个')
    if index != -1:
        print(f"{t} - > {obj['name']}")
