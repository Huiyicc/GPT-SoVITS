# 说明
`fast_api_server.py` 是基于`inference_webui.py` 的封装,内部实现了一个基于`fastapi`的tts合成接口,用于提供模型推理服务。
此文件加入了一个经过训练的情绪7分类的模型(应该只支持中文),用于对输入的文本进行情绪分类。

# 情绪分类
基于bert情绪7分类模型,可以将文本情绪进行打分,将参与训练的lab提前进行批量打分,推理时使用KNN匹配距离最近的向量,得到的就是情绪分至最接近的音频标签,使用这个标签音频作为参考音频输入.

# 启动前配置
- 前往 [modespace](https://modelscope.cn/models/iic/nlp_structbert_emotion-classification_chinese-large/summary) 下载模型
- 放入`GPT_SoVITS/emotion_classification/nlp_structbert_emotion-classification_chinese-large`
- 修改 `数据集情感批量打标.py`
  > 数据集格式参考红血球佬的数据集 (lab -> wav)  
  > https://www.bilibili.com/read/cv26659988/?spm_id_from=333.999.0.0  
    ```python
    # 改成lab集合的路径
    path = r''
    # 打标数据集的输出路径
    out_path = r''
    ```
- 运行`数据集情感批量打标.py`进行情感打标
- 修改`fast_config.json`
  - 修改`gpt_path`为训练的gpt权重路径
  - 修改`sovits_path`为训练的sovits权重路径
  - 修改`model_path`为情感分类模型路径
  - 修改`emotion_data`为打标数据集路径
  - 修改`joblib_path`为KNN模型路径,不存在会自动创建
  - 修改`label_path`为参考音频文本路径
  - 修改`wav_path`为参考音频路径
- 运行`fast_api_server.py`

# API端点
- /v1/voice
  - 参数:
    - text: str -> 输入文本
    - text_language: str -> 输入文本语言
  - 返回:
    - 成功返回音频文件,无本地缓存,需要缓存自行处理

## 其他
- 初版是全分类,但是后来发现对于陈述句和疑问句的识别并不理想,所以预先对句子进行大类分类,再进行情感分类,这样可以提高分类的准确性
- 标点符号会对情感分类产生巨大影响,所以在分类前需要删除所有标点符号