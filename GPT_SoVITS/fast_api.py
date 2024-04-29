import os
import json
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fast_api_server import get_tts_wav


# 初始化配置
class ServerConfig:
    config: dict = {}

    class ConfigServer:
        host: str
        port: int

        def __init__(self, data: dict):
            self.host = data["host"]
            self.port = data["port"]

    class ConfigPrompt:
        ref_wav_path: str
        prompt_text: str
        prompt_language: str

        def __init__(self, data: dict):
            self.ref_wav_path = data["ref_wav_path"]
            self.prompt_text = data["prompt_text"]
            self.prompt_language = data["prompt_language"]

    class ConfigModel:
        bert_path: str
        cnhubert_base_path: str
        gpu_number: int
        gpt_path: str
        sovits_path: str

        def __init__(self, data: dict):
            self.bert_path = data["bert_path"]
            self.cnhubert_base_path = data["cnhubert_base_path"]
            self.gpu_number = data["gpu_number"]
            self.gpt_path = data["gpt_path"]
            self.sovits_path = data["sovits_path"]

    server: ConfigServer
    model: ConfigModel
    prompt: ConfigPrompt

    def __init__(self, cfg: str):
        with open(cfg, "r", encoding="utf-8") as f:
            self.config = json.loads(f.read())
            self.server = self.ConfigServer(self.config["server"])
            self.model = self.ConfigModel(self.config["model"])
            self.prompt = self.ConfigPrompt(self.config["prompt"])


server_config = ServerConfig("fast_config.json")

app = FastAPI()


@app.post("/api/v1/voice")
async def voice_post(request: Request):
    json_data = await request.json()
    text = json_data["text"]
    text_language = json_data["text_language"]
    get_tts_wav(server_config.prompt.ref_wav_path,
                server_config.prompt.prompt_text,
                server_config.prompt.prompt_language,
                text, text_language
                )
    return JSONResponse({"message": f"Hello : 11"})


@app.get("/api/v1/voice")
async def voice_get(name: str):
    return JSONResponse({"message": "Hello World"})


if __name__ == "__main__":
    os.environ["gpt_path"] = server_config.model.gpt_path
    os.environ["sovits_path"] = server_config.model.sovits_path
    os.environ["cnhubert_base_path"] = server_config.model.cnhubert_base_path
    os.environ["bert_path"] = server_config.model.bert_path
    os.environ["_CUDA_VISIBLE_DEVICES"] = str(server_config.model.gpu_number)
    os.environ["is_half"] = str("True")
    uvicorn.run(app, host=server_config.server.host, port=server_config.server.port)
