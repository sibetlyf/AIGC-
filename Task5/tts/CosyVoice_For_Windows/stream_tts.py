import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current)

import gradio as gr
import torch
import json
import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import ffmpeg
import argparse
import itertools

# 读取CosyVoice参数
def read_parameters(log_name:str):
    try:
        with open(log_name, 'r', encoding="utf-8") as f:
            parameters = json.load(f)
        return parameters.get("mode_checkbox_group"), parameters.get("sft_dropdown"), parameters.get("speed_factor"), parameters.get("new_dropdown")
    except FileNotFoundError:
        return None, None, None, None
 # 读取参数
mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown = read_parameters(log_name='./parameters.json')

def set_config():
    sys.path.insert(0, os.path.join(current, 'third_party/Matcha-TTS'))
    sys.path.insert(0, os.path.join(current, 'third_party/AcademiCodec'))
    
    global args, cosyvoice, prompt_sr, target_sr, default_data
    # 默认参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=7863)
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.path.join(current, 'pretrained_models/CosyVoice-300M-Instruct'),
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    # cosyvoice实例
    cosyvoice = CosyVoice(args.model_dir)
    # 默认采样率设置
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    
    # 读取参数
    global mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown
    mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown = read_parameters(log_name=os.path.join(current, './parameters.json'))
    return cosyvoice, default_data, prompt_sr, target_sr, mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown

   
    # return cosyvoice, prompt_sr, target_sr, mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown

# 调用cosyvocie
def start_cosyvoice(text=None):
    global args, current, cosyvoice, prompt_sr, target_sr, default_data
    global mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown
    generator = generate_audio_stream(tts_text=text,
                                mode_checkbox_group=mode_checkbox_group, 
                                sft_dropdown=sft_dropdown, 
                                speed_factor=speed_factor, 
                                new_dropdown=new_dropdown
                            )
    for chunk in generator:
        yield chunk

import time
def stream_out(text=None):
    """
    将输入文本按照两个字拆分，并每隔0.1秒返回一个chunk。
    
    参数:
    text (str): 输入的文本字符串。
    
    返回:
    str: 每个chunk中的文本。
    """
    chunk_size = 2  # 每个chunk的长度
    out = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        out += chunk
        yield out
        time.sleep(0.1)

def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


def generate_audio_stream(tts_text, 
                          mode_checkbox_group=mode_checkbox_group, 
                          sft_dropdown=sft_dropdown, 
                          speed_factor=speed_factor, 
                          new_dropdown=new_dropdown):
    
    global current, cosyvoice, prompt_sr, default_data
    
    if mode_checkbox_group != '预训练音色':
        gr.Warning('流式推理只支持预训练音色推理')
        return (target_sr, default_data)

    spk_id = sft_dropdown

    if new_dropdown != "无":
        spk_id = "中文女"

    joblist = cosyvoice.frontend.text_normalize_stream(tts_text, split=True)

    
    for i in joblist:
        print(i)
        tts_speeches = []
        model_input = cosyvoice.frontend.frontend_sft(i, spk_id)
        if new_dropdown != "无":
            # 加载数据
            print(new_dropdown)
            print("读取pt")
            newspk = torch.load(os.path.join(current, f'./voices/{new_dropdown}.pt'))
            model_input["flow_embedding"] = newspk["flow_embedding"]
            model_input["llm_embedding"] = newspk["llm_embedding"]

            model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
            model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

            model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
            model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

            model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
            model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
            model_input["prompt_text"] = newspk["prompt_text"]
            model_input["prompt_text_len"] = newspk["prompt_text_len"]

        model_output = next(cosyvoice.model.inference_stream(**model_input))
        # print(model_input)
        tts_speeches.append(model_output['tts_speech'])
        output = torch.concat(tts_speeches, dim=1)

        if speed_factor != 1.0:
            try:
                numpy_array = output.numpy()
                audio = (numpy_array * 32768).astype(np.int16) 
                audio_data = speed_change(audio, speed=speed_factor, sr=int(target_sr))
            except Exception as e:
                print(f"Failed to change speed of audio: \n{e}")
        else:
            audio_data = output.numpy().flatten()
            # print(audio_data)
        yield (target_sr, audio_data)


with gr.Blocks() as demo:
    set_config()
    tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")

    audio_output = gr.Audio(label="合成音频",
                            value=None,
                            streaming=True,
                            autoplay=True,  # disable auto play for Windows, due to https://developer.chrome.com/blog/autoplay#webaudio
                            interactive=False,
                            show_label=True,
                            show_download_button=True)
    generate_button_stream = gr.Button("生成音频")
    generate_button_stream.click(
                    fn=stream_out,
                    inputs=[tts_text],
                    outputs=[tts_text]
                    ).then(start_cosyvoice,
                    inputs=[tts_text],
                    outputs=[audio_output],)



if __name__ == "__main__":
    set_config()
    global args
    # 启动gradio应用
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_port=args.port, inbrowser=True)