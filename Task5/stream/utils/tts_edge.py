import json
import logging
import os
import sys
from gradio_client import Client
import traceback
import asyncio
import numpy as np
current_path = os.path.dirname(os.path.abspath(__file__))

import edge_tts


async def get_data(text):
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        voice_tmp_path = loop.run_until_complete(edgetts(text))
    finally:
        return voice_tmp_path
        loop.close()

# edgetts直接生成的wav格式音频中与常见格式不同，包括type与data两个数据，不能直接用sondfile等库转换为np格式
# 而且stream方式中是一个异步协程函数， 没法直接迭代
async def edgetts(messages, VOICE="zh-CN-XiaoxiaoNeural"):
    try:
        VOICE = VOICE
        TEXT = messages
        # 初始化文本转语音的通信对象
        communicate = edge_tts.Communicate(TEXT, VOICE)

        # 确定输出文件名称
        file_path = os.path.join(current_path, 'voice_tmp')
        os.makedirs(file_path, exist_ok=True)
        file_name = f'temp.wav'
        voice_tmp_path = os.path.join(file_path, file_name)
        
        # 打开文件以便写入
        with open(voice_tmp_path, "wb") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
                    # print(f"WordBoundary: {chunk}")

                # elif resp["type"] == "WordBoundary":
                #     print(resp)  # 打印元数据信息
                # else:
                #     print(resp)

        return voice_tmp_path
    
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(f'TTS请求失败: {e}')
        return None

# 二进制转np数组
def bytes_to_numpy(audio_bytes):
    # 默认采样率设置
    target_sr = 22050
    audio_data = np.zeros(3, dtype=np.int16)
    # 将二进制数据转换为NumPy数组
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_np = audio_np.flatten()
    print(len(audio_np), audio_data.mean())
    audio_data = np.concatenate([audio_data, audio_np])
    return (target_sr, audio_data)

def get_result(text):
    result = asyncio.run(edgetts(text))
    return result

if __name__ == '__main__':
    import gradio as gr
    text = "你好，我是移动鸿鹄训练营第三小组设计的数字人助手，有什么可以帮你的吗？"
    # 使用 asyncio.run 来运行异步函数
    # with gr.Blocks() as demo:
    #     text = gr.Textbox(label="输入文本")
    #     audio = gr.Audio(label="语音", type='numpy')
    #     submit_button = gr.Button("生成语音")
    #     submit_button.click(get_data, inputs=[text], outputs=[audio])
    # demo.launch()
    print(get_result(text))
    