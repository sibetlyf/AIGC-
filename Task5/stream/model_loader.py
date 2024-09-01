from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn
# 异步库
import asyncio
# 线程池执行器
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import TextIteratorStreamer
import torch

import json
import datetime
import os
import sys
import logging
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

global model_path
model_path = os.path.join(current_dir, "../models/qwen2")

# 创建fastapi
app = FastAPI()

logging.basicConfig(level=logging.DEBUG)


def load_model_and_tokenizer(model_path):
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # 加载预训练模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  # 允许动态量化
            ).eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or rokenizer: {e}")
        raise

# 大模型推理
async def LLM_carry(prompt):
    global model_path
    global model, tokenizer # 声明全局变量，在函数内部使用模型和分词器
    tokenizer, model = load_model_and_tokenizer(model_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    torch.cuda.empty_cache() # 执行GPU内存清理
    
    # 使用分词器的apply_chat_template方法格式化消息
    input_ids = tokenizer.apply_chat_template(
        messages, # 待格式化的消息
        tokenize=False, # 不进行分词
        add_generation_prompt=True, # 添加生成提示
    )
    
    # 将格式化后的文本转换为模型驶入，并转换为张量，然后移动到指定设备
    model_inputs = tokenizer(
        [input_ids],
        return_tensors="pt",
    ).to("cuda")
    
    # 使用流式传输模型
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
    )
    
    # 使用model.generate()方法一次响应生成文本
    generated_ids = model.generate(
        model_inputs.input_ids, # 模型驶入的input_ids
        max_new_tokens=4096, # 最大新生成的token数量
        streamer=streamer,
    )
    # 执行流式输出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # FastAPI的流式响应
    async def stream_response():
        for text in streamer:
            if text:
                #print(repr(text))
                # 用yield迭代推送对话内容
                # StreamingResponse下的返回信息
                #yield text
                # EventSourceResponse下的返回信息
                yield {
                    "event": "message",
                    "retry": 15000,
                    "data": repr(text)  # 防止\n换行符等传输过程中丢失
                }
 
    #return StreamingResponse(stream_response(), media_type="application/octet-stream")
    return EventSourceResponse(stream_response())

@app.post("/generate")
async def generate_text(prompt:str):
    return await LLM_carry(prompt)


def main():
    # 启动fastapi应用
    uvicorn.run(app, host="localhost", port=12345)

if __name__ == "__main__":
    main()