from KNN模型 import find_similar_sentences

if __name__ == '__main__':
    # tests = [
    #     "进攻D点！",
    #     "不行！",
    #     "坚持一下胜利就是我们的了！",
    #     "攻击敌军基地！",
    #     "我们的局势不太妙！",
    #     "胜败乃兵家常事，但是下一次我们会赢回来的。",
    #     "那么，代价是什么？",
    #     "在这无尽的思念中，我终于明白，你就是我生命的全部。",
    #     ]
    # for test in tests:
    #     print(test, " ->", find_similar_sentences(test))
    while True:
        text = input("请输入文本：")
        print(find_similar_sentences(text))
        print()

