import json

from starlette.responses import StreamingResponse

from fast_config_model import app_config
from emotion_classification.KNN模型 import find_similar_sentences

from inference_webui import get_tts_wav, i18n
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from scipy.io.wavfile import write as wav_write
from io import BytesIO
import numpy as np

emotion_classification_path = r"GPT_SoVITS/emotion_classification/nlp_structbert_emotion-classification_chinese-large"

app = FastAPI()


class TTSConfig(BaseModel):
    text: str
    text_language: str


@app.post("/api/v1/voice")
async def create_item(cfg: TTSConfig):
    # 预测情感分数
    scores, sentence = find_similar_sentences(cfg.text)
    # 匹配预测文件
    prompt_file = app_config.emotion_classification.wav_path + "/" + sentence[0]["file"] + ".wav"
    prompt_content = sentence[0]["content"]
    print(f"预测文件：{prompt_file}, 预测内容：{prompt_content}")
    # 生成音频
    res = get_tts_wav(prompt_file, prompt_content, "中英混合", cfg.text, cfg.text_language, i18n("凑四句一切"), 5, 1, 1)
    sample_rate, audio_data = next(res)
    audio_bytes = BytesIO()
    wav_write(audio_bytes, sample_rate, audio_data.astype(np.int16))
    headers = {
        "Content-Disposition": "attachment; filename=output_audio.wav",
    }
    return StreamingResponse(audio_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18080)
