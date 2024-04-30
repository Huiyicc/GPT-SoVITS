import json
import os


class ServerConfig:
    def __init__(self, host, port):
        self.host = host
        self.port = port

class ModelConfig:
    def __init__(self, bert_path, cnhubert_base_path, gpu_number, gpt_path, sovits_path):
        self.bert_path = bert_path
        os.environ["bert_path"] = bert_path
        self.cnhubert_base_path = cnhubert_base_path
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        self.gpu_number = gpu_number
        os.environ["gpu_number"] = gpu_number
        self.gpt_path = gpt_path
        os.environ["gpt_path"] = gpt_path
        self.sovits_path = sovits_path
        os.environ["sovits_path"] = sovits_path

class EmotionClassificationConfig:
    def __init__(self, model_path, emotion_data, joblib_path, label_path, wav_path):
        self.model_path = model_path
        os.environ["EMOTION_MODEL_FILE"] = model_path
        self.emotion_data = emotion_data
        os.environ["EMOTION_DATA_FILE"] = emotion_data
        self.joblib_path = joblib_path
        os.environ["EMOTION_JOBLIB_FILE"] = joblib_path
        self.label_path = label_path
        os.environ["EMOTION_LABEL_FILE"] = label_path
        self.wav_path = wav_path

class AppConfig:
    def __init__(self, server:ServerConfig, model:ModelConfig, emotion_classification:EmotionClassificationConfig):
        self.server = server
        self.model = model
        self.emotion_classification = emotion_classification


with open("GPT_SoVITS/fast_config.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为类实例
server_config = ServerConfig(data['server']['host'], data['server']['port'])
model_config = ModelConfig(
    data['model']['bert_path'],
    data['model']['cnhubert_base_path'],
    data['model']['gpu_number'],
    data['model']['gpt_path'],
    data['model']['sovits_path']
)
emotion_classification_config = EmotionClassificationConfig(
    data['emotion_classification']['model_path'],
    data['emotion_classification']['emotion_data'],
    data['emotion_classification']['joblib_path'],
    data['emotion_classification']['label_path'],
    data['emotion_classification']['wav_path']
)

app_config = AppConfig(server_config, model_config, emotion_classification_config)