from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydantic import BaseModel
import base64
from fastapi import FastAPI
import uvicorn
import requests
import json

url = "http://127.0.0.1:8000/"
# 客户端使用ASR
def as_api_client(wav_path):
    headers = {'Content-Type': 'application/json'}
    with open(wav_path, "rb") as f:
        wav = base64.b64encode(f.read()).decode()
    data = {"wav": wav}
    url = "http://127.0.0.1:8000/"
    response = requests.post(url+"asr", headers=headers, json=data)
    response = response.json()
    if response['code'] == 0:
        res = response['res']
        return res
    else:
        return response['msg']
