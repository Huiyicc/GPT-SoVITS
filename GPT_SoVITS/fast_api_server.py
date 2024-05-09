import json
import os

import re
import pandas as pd
import torch


class ReCut():
    def __init__(self, text):
        self.text = text
        self.text =self.text.replace("...","{{spend_split}}")

    def cut(self):
        # l = self.__connect__(self.text)
        # res = [""]
        # index = 0
        # pattern = r'[。？！；……~.?;!]'
        # for i in l:
        #     for j in i:
        #         res[index] += j
        #         if re.search(pattern, j):
        #             res.append("")
        #             res[index] = res[index].replace("{{spend_split}}", "...")
        #             index += 1
        # return res
        return self.__connect__(self.text)

    def __cut__(self, para):
        pattern = [r'([。！？\?])([^”\'])', r'(\.{6})([^”\'])', r'(\…{2})([^”\'])', r'([。！？\?][”\'])([^，。！？\?])']
        for i in pattern:
            para = re.sub(i, r"\1\n\2", para)
        para = para.rstrip()
        return para.split("\n")

    def __connect__(self, paragraph):
        paragraph = paragraph.replace("“", "\"")
        paragraph = paragraph.replace("”", "\"")
        sentences = self.__cut__(paragraph)
        # for each_para in paragraph:
        #     sentence_before.append(self.__cut__(each_para))

        result_list = []

        for sentence in sentences:
            sslit = sentence.split("\"")
            for i in sslit:
                if i.strip() == "":
                    continue
                result_list.append(i.replace("{{spend_split}}", "..."))

        return result_list

from starlette.responses import StreamingResponse

from fast_config_model import app_config
from emotion_classification.KNN模型 import find_similar_sentences

from inference_webui import get_tts_wav, i18n, get_first, cut1, cut2, cut3, cut4, cut5,hps
from inference_webui import splits as web_splits
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from scipy.io.wavfile import write as wav_write
from io import BytesIO
import numpy as np

emotion_classification_path = r"GPT_SoVITS/emotion_classification/nlp_structbert_emotion-classification_chinese-large"

app = FastAPI()

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()


class TTSConfig(BaseModel):
    text: str
    text_language: str
    split_sentence: int
@app.post("/v1/voice")
async def voice(cfg: TTSConfig):
    text_language = cfg.text_language
    text = cfg.text.strip("\n")

    if (text[0] not in web_splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    if (cfg.split_sentence == 1):
        text = cut1(text)
    elif (cfg.split_sentence == 2):
        text = cut2(text)
    elif (cfg.split_sentence == 3):
        text = cut3(text)
    elif (cfg.split_sentence == 4):
        text = cut4(text)
    elif (cfg.split_sentence == 5):
        text = cut5(text)
    text_list = ReCut(text).cut()
    audio_segments = []
    sample_rate = 0
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    for text_i in text_list:
        text_i = text_i.strip()
        if len(text_i) == 0:
            continue
        # 预测情感分数
        scores, sentence = find_similar_sentences(text_i)
        # 匹配预测文件
        prompt_file = app_config.emotion_classification.wav_path + "/" + sentence[0]["file"] + ".wav"
        prompt_content = sentence[0]["content"]
        print(f"预测文件：{prompt_file}, 预测内容：{prompt_content}")
        # 生成音频
        res = get_tts_wav(prompt_file, prompt_content, "中英混合", text_i, cfg.text_language, i18n("不切"), 5, 1, 1)
        sample_rate, audio_data = next(res)
        # audio_bytes_segment = BytesIO()
        # wav_write(audio_bytes_segment, sample_rate, audio_data.astype(np.int16))
        # audio_bytes_segment.seek(0)  # 将指针移到开头，以便后续读取
        audio_segments.append(audio_data)
        audio_segments.append(zero_wav)
    m_data = np.concatenate(audio_segments)
    final_audio_bytes = BytesIO()
    wav_write(final_audio_bytes, sample_rate, m_data.astype(np.int16))

    final_audio_bytes.seek(0)

    headers = {
        "Content-Disposition": "attachment; filename=output_audio.wav",
    }
    return StreamingResponse(final_audio_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host=app_config.server.host, port=app_config.server.port)
