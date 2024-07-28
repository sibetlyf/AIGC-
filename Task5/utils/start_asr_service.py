from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import base64

url = "http://127.0.0.1:8000/"

# asr model
model_dir = r"../ASR"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)


# 定义asr数据模型，用于接收POST请求中的数据
class ASRItem(BaseModel):
    wav: str  # 输入音频


app = FastAPI()


@app.post("/asr")
async def asr(item: ASRItem):
    try:
        data = base64.b64decode(item.wav)
        with open("test.wav", "wb") as f:
            f.write(data)
        res = model.generate("test.wav",
                             language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                             use_itn=True,
                             batch_size_s=60,
                             merge_vad=True,  #
                             merge_length_s=15, )
        text = rich_transcription_postprocess(res[0]["text"])
        result_dict = {"code": 0, "msg": "ok", "res": text}
    except Exception as e:
        result_dict = {"code": 1, "msg": str(e)}
    return result_dict

if __name__ == "__main__":
    config_path = '../config.json'
    # 读取json
    with open(config_path, 'r', encoding='utf-8') as file:
        config_content = file.read()
    config = json.loads(config_content) # 将JSON字符串转换为字典
    uvicorn.run(app, host=config['ASR']['ip'], port=config['ASR']['port'])