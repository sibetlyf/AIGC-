import gradio as gr
import logging
import json
import numpy as np
import asyncio
import os
import sys
import time
import webbrowser
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# local
from request_post import ChatGPT

# 直接返回
def return_directly(*args):
    return args

# 读取日志文件
from utils.config import Config, textarea_data_change
config_path = os.path.join(current_path, "configs", "config.json")
config = Config(config_path)

global config_data
def load_config(config_path= os.path.join(current_path, "configs", "config.json")):
    global config_data
    try:
        with open(config_path, 'r', encoding="utf-8") as config_file:
            config_data = json.load(config_file)
            return ("成功加载配置文件")
    except Exception as e:
        logging.error(f"无法读取配置文件！\n{e}")
        
load_config(config_path)


    
# 保存配置
def save_config(api, api_key, model, temperature, max_tokens, top_p,  
                tts_api_ip_port, tts_model, edgetts_voice_dropdown,
                gradio_save_local):
    global config_data

    try:
        with open(config_path, 'r', encoding="utf-8") as config_file:
            config_data = json.load(config_file)
    except Exception as e:
        logging.error(f"无法读取配置文件！\n{e}")
        gr.Error(f"无法读取配置文件！\n{e}")
        return f"无法读取配置文件！{e}"
    
    def common_textarea_handle(content):
        """通用的textEdit 多行文本内容处理

        Args:
            content (str): 原始多行文本内容

        Returns:
            _type_: 处理好的多行文本内容
        """
        # 通用多行分隔符
        separators = [" ", "\n"]

        ret = [token.strip() for separator in separators for part in content.split(separator) if (token := part.strip())]
        if 0 != len(ret):
            ret = ret[1:]

        return ret
    
    config_data["openai"]["api"] = api
    config_data["openai"]["api_key"] = common_textarea_handle(api_key)
    config_data["chatgpt"]["model"] = model
    config_data["chatgpt"]["temperature"] = float(temperature)
    config_data["chatgpt"]["max_tokens"] = int(max_tokens)
    config_data["chatgpt"]["top_p"] = float(top_p)
    config_data["tts"]["tts_ip_port"] = tts_api_ip_port
    config_data["tts"]["model"] = tts_model
    config_data["tts"]["edgetts_voice_dropdown"] = edgetts_voice_dropdown

    # 重载chatgpt
    # chatgpt = Chatgpt(config_data["openai"], config_data["chatgpt"])
    
    if False == gradio_save_local:
        logging.info("配置已加载")
        return "配置已加载"

    # 写入配置到配置文件
    try:
        with open(config_path, 'w', encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=2, ensure_ascii=False)
            config_file.flush()  # 刷新缓冲区，确保写入立即生效

        logging.info("配置数据已成功写入文件！")
        gr.Info("配置数据已成功写入文件！")

        return "配置数据已成功写入文件！"
    except Exception as e:
        logging.error(f"无法读取配置文件！\n{e}")
        gr.Error(f"无法读取配置文件！\n{e}")
        return f"无法读取配置文件！{e}"



## ChatGPT功能实现
def get_response(prompt, messages, encrypt=None):
    print('开始接受输入\n', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    global config_data
    chat_bot = ChatGPT(api_key=config_data['openai']['api_key'], api=config_data['openai']['api'])
    chat_bot.reset_configs(config_data)
    response = chat_bot.chat(prompt, messages)
    resp = ""
    try:
        # 处理流式响应
        for i, chunk in enumerate(response):
            if chunk:
                try:
                    resp += chunk
                    # 输出模型返回的内容
                    print(i, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), len(chunk))
                    # print(resp) 
                    yield resp
                except ValueError:
                    print("Invalid data")
    except KeyboardInterrupt:
        pass
    finally:
        print(resp, "\n对话结束",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), len(chunk))
        response.close()

## ASR功能实现
from scipy.io import wavfile
from request_post import asr_api_client
from utils.common import Common
import librosa

common = Common()
def get_asr_result(audio):
    print("开始识别")
    if isinstance(audio, str): # 如果传入音频地址
        # 获取语音识别结果
        stt_ret = asr_api_client(audio)
        logging.info(f"语音识别内容：{stt_ret}")
        yield stt_ret
    else: # 如果传入（sr， np）格式音频
        try:
            # 创建一个示例的 int16 音频数据
            int16_audio_data = np.array(audio[1], dtype=np.int16)

            # 创建一个临时文件来存储录制的音频数据
            output_file = "out/" + common.get_bj_time(4) + ".wav"

            # 使用 scipy.io.wavfile 将数据保存为 WAV 文件
            wavfile.write(output_file, audio[0], int16_audio_data)
            
            # 获取语音识别结果
            stt_ret = asr_api_client(output_file)
            logging.info(f"语音识别内容：{stt_ret}")
            yield stt_ret
        except Exception as e:
            logging.error(f"语音识别失败！\n{e}")
            return f"语音识别失败！\n{e}"

# TTS功能实现
parent_path = os.path.dirname(current_path)
tts_path = os.path.join(parent_path, "tts\CosyVoice_For_Windows")
sys.path.append(parent_path)
from tts.CosyVoice_For_Windows import stream_tts
from stream.utils.tts_edge import edgetts
import torch
import numpy as np
import ffmpeg
import random

## tts相关变量
cosyvoice, default_data, prompt_sr, target_sr, mode_checkbox_group, sft_dropdown, speed_factor, new_dropdown = stream_tts.set_config()
## zero-shot后的flow，保存在..\tts\CosyVoice_For_Windows\voices文件夹下
spk_path = os.path.join(parent_path, 'tts/CosyVoice_For_Windows/voices')
## 读取现有zero-shot复刻角色
spk_new = ["无"]
for name in os.listdir(spk_path):
    gr.Info(f'检测到复刻角色{name.replace(".pt","")}')
    spk_new.append(name.replace(".pt",""))
# 刷新复刻角色列表
def get_spk_new(spk_path=spk_path):

    spk_new = ["无"]
    for name in os.listdir(spk_path):
        spk_new.append(name.replace(".pt",""))
        
    return {"choices":spk_new, "__type__": "update"}

## 语音复刻相关函数
import librosa
import shutil
import torchaudio

### 音频加载
def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

### 声音预处理
max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

### 零样本复刻
def zeroshot(tts_text, use_zero_shot, sft_dropdown, prompt_text, prompt_wav_upload, new_name, new_dropdown):
    
    global current, cosyvoice, prompt_sr, default_data
    global config_data # 配置数据
    
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    else:
        prompt_wav = None
    
    if use_zero_shot == "是":
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)

        # 防止无prompt文本
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (target_sr, default_data)

        # 开始复刻
        seed = 0
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    else:
        # 直接使用预训练模型推理
        seed = 0
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, sft_dropdown, new_dropdown)
    
    output = output['tts_speech']
    # 转换为int16
    if speed_factor != 1.0:
        try:
            numpy_array = output.numpy()
            audio = (numpy_array * 32768).astype(np.int16) 
            audio_data = speed_change(audio, speed=speed_factor, sr=int(target_sr))
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
    else:
        audio_data = output.numpy().flatten()


    return (target_sr, audio_data)

        
        

### 保存零样本推理结果
def save_name(name):
    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False
    origin_file = os.path.join(parent_path, 'output.pt')
    target_file = os.path.join(parent_path, f'tts/CosyVoice_For_Windows/voices/{name}.pt')
    shutil.copyfile(origin_file, target_file)
    gr.Info(f"音色保存成功,存放位置为{target_file}目录")
    return target_file

## tts相关函数
def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

stop_falg = False
def stop_stream():
    global stop_falg
    stop_falg = True

## 流式生成函数
def generate_audio_stream(tts_text,
                          sft_dropdown=sft_dropdown, 
                          new_dropdown=new_dropdown,
                          speed_factor=speed_factor):
    
    global current, cosyvoice, prompt_sr, default_data
    global config_data # 配置数据
    global stop_falg # 停止标记，用于停止流式输出
    stop_falg = False


   
    spk_id = sft_dropdown

    if new_dropdown != "无":
        spk_id = "中文女"

    joblist = cosyvoice.frontend.text_normalize_stream(tts_text, split=True)

    
    for i in joblist:
        if stop_falg==True:
            break
        print(i)
        tts_speeches = []
        model_input = cosyvoice.frontend.frontend_sft(i, spk_id)
        if new_dropdown != "无":
            # 加载数据
            print(new_dropdown)
            print("读取pt")
            newspk = torch.load(os.path.join(tts_path, f'./voices/{new_dropdown}.pt'))
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

        yield (target_sr, audio_data)

## 情感控制函数
from request_post import get_instrut

def get_text_instruct(text):
    instruct, emo = get_instrut(text)
    return instruct, emo

def generate_audio_instruct(tts_text, 
                            instruct_text,
                            sft_dropdown=sft_dropdown, 
                            new_dropdown=None,
                            speed_factor=speed_factor, 
                            seed=0,
                            target_sr=22050,
                            ):
    if cosyvoice.frontend.instruct is False:
        gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M-Instruct模型'.format(args.model_dir))
        return (target_sr, default_data)
    if instruct_text == '':
        gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
        return (target_sr, default_data)

    try:
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, new_dropdown)
        audio_data = output['tts_speech'].numpy().flatten()
        return (target_sr, audio_data)
    except Exception as e:
        print(f"Failed to generate audio: \n{e}")
        return (target_sr, default_data)

# RAG功能实现
from RAG.indexer import indexe_pdf
from request_post import get_rag_response

# pdf转向量
def pdf2vec(pdf_path):
    """
    将PDF文件转换为向量表示。
    :param pdf_path: PDF文件的路径
    :return: 如果PDF文件存在且转换成功，则返回True；否则返回False
    """
    # 验证路径的有效性
    if not os.path.exists(pdf_path):
        print("Error: The specified PDF file does not exist.")
        return False
    
    # 调用具体的转换逻辑
    try:
        faiss_path = indexe_pdf(pdf_path=pdf_path)
        return faiss_path
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False
    
def get_rag_resp(faiss_path, messages):
    # 验证路径有效性
    if not os.path.exists(faiss_path):
        print("Error: The faiss file dose not exist!")
        return False
    
    # 调用API
    resp = get_rag_response(faiss_path=faiss_path, messages=messages)
    return resp

# Router Chain功能实现
from utils.Router_chain import RouterChainManager
from request_post import get_router_response

agents = ["doctor", "secretary", "english_teacher"]
## 获取自定义prompt
def get_custom_agent():
    global agents
    manager = RouterChainManager()
    custom_agents = manager.load_custom_agents()
    for name, data in custom_agents.items():
        agents.append(name)
    return agents

## 自定义增加与删除prompt
def create_agent(name, description, template):
    manager = RouterChainManager()
    manager.save_custom_agent(name=name, description=description, template=template)
    gr.Info('添加成功！')
    return True

def delete_agent(name):
    manager = RouterChainManager()
    manager.delete_custom_agent(name=name)
    gr.Info('删除成功！')
    return True

agents_used = []
def check_agents(*args):
    """根据输入确定使用的agents类型，并将其名称写入agents_used"""
    global custom_agents
    # 验证args是否为空
    if not args:
        gr.Info("未选择agents，采用默认agent")
        return False  # 如果args为空，则直接返回False
    # 初始化agents_used
    agents_used.clear()

    # 假设args中只包含一个列表
    input_agents = args[0]

    # 根据输入参数确定使用的agents
    for agent in input_agents:
        if agent == "doctor":
            agents_used.append("doctor")
        elif agent == "secretary":
            agents_used.append("secretary")
        elif agent == "english_teacher":
            agents_used.append("english_teacher")
    if len(input_agents) > 3:
        agents_used.append("custom_agents")
    gr.Info("成功更新agents！")
    return True

## 获取输出
def get_router_chain_response(query,):
    global custom_agents
    # 解析agents_used以获取实际使用的代理
    doctor = "doctor" in agents_used
    secretary = "secretary" in agents_used
    english_teacher = "english_teacher" in agents_used
    if "custom_agents" in agents_used:
        custom_agents = True
    else:
        custom_agents = False
    
    resp = get_router_response(query=query, doctor=doctor, secretary=secretary, english_teacher=english_teacher, custom_agents=custom_agents)
    return resp

# 插入视频
def return_video_path():
    return os.path.join(current_path, r'MimicMotion/outputs/talker.mp4')

def main():
    with gr.Blocks() as demo:
        with gr.Tab("流对话模式") as tab1:
            with gr.Row(variant="compact"):
                instruction_text_1 = gr.Text(label="操作步骤", 
                                           value="1.输入音频（可选）或文字，ASR识别结果会自动填充到文本框\n2.选择音源，点击提交按钮，开始对话\n3.对话结果会自动转为音频并流式播放",
                                           )
            with gr.Row():
                au = gr.Audio(label="输入音频进行识别",
                            interactive=True,
                            show_label=True,
                            show_download_button=True)
                with gr.Column():
                    msg = gr.Textbox(
                        label="输入内容",
                        placeholder="请告诉我你想要了解的问题",
                        show_label=True,
                        lines=10,
                        interactive=True,
                        scale=10,
                    )
                    prompt = gr.Textbox(
                        label="prompt",
                        value="你是移动鸿鹄训练营第三小组设计的数字人助手，能够在200字以内精确回答用户提问",
                        show_label=True,
                        lines=1,
                        interactive=True,
                        scale=1
                    )
                    with gr.Row():
                        submit_button = gr.Button("发送", variant='primary')
                    au.change(fn=get_asr_result, inputs=[au], outputs=[msg])
            with gr.Row():
                resp_display = gr.Textbox(label="Bot回复")
                # 当点击发送按钮时，获取响应并播放音频
                with gr.Group():
                    with gr.Column():
                        # 预训练音色
                        sft_spk = cosyvoice.list_avaliable_spks()
                        sft_dropdown = gr.Dropdown(label="选择预训练音色", choices=sft_spk, value=sft_spk[0], interactive=True,)
                        new_dropdown = gr.Dropdown(label='选择新增音色', choices=spk_new, value=spk_new[0],interactive=True)
                        refresh_button = gr.Button("刷新参考音频")
                    audio_output = gr.Audio(label="合成音频",
                                            value=None,
                                            streaming=True,
                                            autoplay=True,  # disable auto play for Windows, due to https://developer.chrome.com/blog/autoplay#webaudio
                                            interactive=False,
                                            show_label=True,
                                            show_download_button=True)
                    video_output = gr.Video(autoplay=True, label="视频", value=None, interactive=False, loop=True, max_length=500)
                    stop_button = gr.Button("停止对话", variant='stop')
                submit_button.click(
                fn=get_response,
                inputs=[prompt, msg],
                outputs=[resp_display],
                ).then(
                    fn=return_video_path,
                    inputs=[],
                    outputs=[video_output],
                ).then(
                    fn=generate_audio_stream,
                    inputs=[resp_display, sft_dropdown, new_dropdown],
                    outputs=[audio_output],
                )
                stop_button.click(fn=lambda: (None, None), 
                                  inputs=[], 
                                  outputs=[audio_output]
                                  ).then(
                                      fn=stop_stream,
                                      inputs=None,
                                      outputs=None
                                  )
            
        # 页面2
        with gr.Tab("情感控制模式") as tab2:
            with gr.Row(variant="compact"):
                instruction_text_2 = gr.Text(label="操作步骤", 
                                        value="1.输入文本\n2.选择音源，点击提交按钮，开始对话\n3.模型会自动生成文本情感instruct，并在文本框中显示\n4.点击生成按钮，生成音频",
                                        )
            with gr.Row():
                with gr.Column():
                    # 预训练音色
                    sft_spk = cosyvoice.list_avaliable_spks()
                    sft_dropdown = gr.Dropdown(
                        label="选择预训练音色",
                        choices=sft_spk,
                        value=sft_spk[0],
                        interactive=True,
                    )
                    new_dropdown = gr.Dropdown(choices=spk_new, label='选择新增音色', value=spk_new[0], interactive=True)
                    msg = gr.Textbox(
                        label="输入需要转换的文本",
                        value="我是鸿鹄梧桐训练营第三组设计的数字人，提供舒适自然的语音合成能力",
                        show_label=True,
                        lines=10,
                        interactive=True,
                        scale=10,
                    )
                    submit_button = gr.Button("获取instruct与emotion标记", variant='primary')
                with gr.Column():
                    msg_new = gr.Textbox(
                        label="加入情感标签后的文本",
                        value="",
                        show_label=True,
                        interactive=False,
                        scale=10,
                    )
                    instruct = gr.Textbox(
                        label="instruct",
                        value="",
                        show_label=True,
                        interactive=True,
                        scale=5,
                    )
                    submit_button.click(
                        fn = get_text_instruct,
                        inputs=[msg],
                        outputs=[instruct, msg_new]
                    )
                    audio_generate = gr.Button("生成音频", variant='primary')
                    
            with gr.Column():
                audio_out = gr.Audio(label="合成音频",
                                    value=None,
                                    streaming=True,
                                    autoplay=True,
                                    interactive=False,
                                    show_label=True,
                                    show_download_button=True,
                                    min_width=100,
                                    )
                audio_generate.click(
                    fn=generate_audio_instruct,
                    inputs=[msg_new, instruct, sft_dropdown, new_dropdown],
                    outputs=[audio_out]
                )
        # 页面3
        with gr.Tab("RAG模式") as tab3:
            with gr.Row(variant="compact"):
                instruction_text = gr.Text(label="操作步骤", 
                                           value="1.上传需要检索的文本,点击提交按钮\n2.输入需要检索的内容\n3.点击检索按钮，模型会自动检索文本，并在文本框中显示",
                                          )
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        upload_file = gr.File(label="上传文件", type="filepath", file_types=['.pdf'], interactive=True, scale=1)
                        submit_button = gr.Button("提交", variant='primary')
                        faiss_path = gr.Textbox(label="向量库地址", 
                                                value=os.path.join(current_path,r"RAG/vectorbase/LLM.faiss"),
                                                show_label=True, 
                                                interactive=True, 
                                                scale=1)
                        submit_button.click(
                            fn=pdf2vec,
                            inputs=[upload_file],
                            outputs=[faiss_path]
                        )
                    msg = gr.Textbox(
                        label="输入内容",
                        placeholder="请告诉我你想要了解的问题",
                        show_label=True,
                        lines=10,
                        interactive=True,
                        scale=20,
                    )
                    chat_button = gr.Button("开始检索", variant='primary')
                with gr.Column():
                    resp = gr.Textbox(label="Bot回复", 
                                      show_label=True,
                                      interactive=False)
                    audio_out = gr.Audio(label="合成音频",
                                    value=None,
                                    streaming=True,
                                    autoplay=True,
                                    interactive=False,
                                    show_label=True,
                                    show_download_button=True,
                                    )
                    chat_button.click(
                        fn=get_rag_resp,
                        inputs=[faiss_path, msg],
                        outputs=[resp],
                        queue=False
                    )
        # 页面4：Router Chain
        with gr.Tab("多Agent模式") as tab4:
            with gr.Row(variant="compact"):
                instruction_text = gr.Text(label="操作步骤", 
                                           value="1.选择需要的agent,点击提交按钮\n2.输入提问内容\n3.点击提交按钮，模型会自动选择合适的agent",
                                           )
                get_custom_agent()
                check_agents(agents)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        custom_name = gr.Textbox(
                            label="自定义角色名称", 
                            placeholder="给我取个名字吧！", 
                            show_label=True,
                            interactive=True,)
                        custom_description = gr.Textbox(
                            label="自定义角色的功能描述",
                            placeholder="给我想个身份吧！",
                            show_label=True,
                            interactive=True,
                        )
                    with gr.Row():
                        custom_prompt = gr.Textbox(
                            label="自定义prompt",
                            placeholder="请输入自定义prompt",
                            show_label=True,
                            interactive=True,
                        )
                    submit_prompt = gr.Button("提交", variant='primary')
                    submit_prompt.click(fn=create_agent, inputs=[custom_name, custom_description, custom_prompt], outputs=None)
                    with gr.Row():
                        a = gr.Textbox(visible=False, scale=3)
                    with gr.Row():
                        delete_name = gr.Textbox(
                            label="请输入想要删除的角色",
                            show_label=True,
                            interactive=True,
                        )
                        delete_button = gr.Button("删除")
                        delete_button.click(fn=delete_agent, inputs=[delete_name], outputs=None)
                    
            with gr.Column():
                choose_agents = gr.CheckboxGroup(
                    label="选择agent",
                    choices=agents,
                    value=agents,
                    interactive=True,
                    scale=3,
                )
                refresh_button = gr.Button("刷新", variant='primary')
                refresh_button.click(fn=get_custom_agent, inputs=None, outputs=None)
                    # .then(
                    #     lambda: gr.update(choices=agents_used, value=agents_used), None, choose_agents,
                    # )
                choose_agents.change(fn=check_agents, inputs=[choose_agents], outputs=None)
            with gr.Row():
                with gr.Column():
                    msg = gr.Textbox(
                        label="输入内容",
                        placeholder="请告诉我你想要了解的问题",
                        show_label=True,
                        lines=10,
                        interactive=True,
                    )
                    chat_button = gr.Button("提交", variant='primary')
                resp = gr.Textbox(label="Bot回复", 
                                  show_label=True,
                                  interactive=False)
                chat_button.click(
                    fn=get_router_chain_response,
                    inputs=[msg],
                    outputs=[resp]
                )
        # 页面5：语音复刻
        with gr.Tab("语音合成") as tab5:
            with gr.Row(variant="compact"):
                zeroshot_text = gr.Text(label="操作步骤", 
                                        value="1.选择需要复刻的音源（30s内）\n2.输入与音源对应的文本（ASR会自动检测，请核对内容）\n3.提交需要生成的文本内容\n4.点击生成按钮，生成复刻音频",
                                        max_lines=100)
            with gr.Row():
                with gr.Column():
                    # 预训练音色
                    sft_spk = cosyvoice.list_avaliable_spks()
                    sft_dropdown = gr.Dropdown(
                        label="选择预训练音色",
                        choices=sft_spk,
                        value=sft_spk[0],
                        interactive=True,
                    )
                    use_zero_shot = gr.Radio(
                        label="是否使用语音复刻功能",
                        choices=["是", "否"],
                        value = '否',
                    )
                    new_dropdown = gr.Dropdown(choices=spk_new, label='选择新增音色', value=spk_new[0],interactive=True)
                    refresh_button = gr.Button("刷新参考音频")
                
                tts_text = gr.Textbox(
                    label="输入合成文本", 
                    lines=3, 
                    value="我是移动鸿鹄第三组全新推出的生成式数字人，提供舒适自然的交互能力。")
            with gr.Row():
                with gr.Column():
                    prompt_wav_upload = gr.Audio(label="输入prompt音频",
                            interactive=True,
                            show_label=True,
                            show_download_button=True,
                            type='filepath')
                    prompt_asr_button = gr.Button("开始识别prompt语音")
                prompt_msg = gr.Textbox(
                    label="输入对应的prompt文本",
                    placeholder="请输入prompt文本，需与prompt音频内容一致",
                    show_label=True,
                    lines=10,
                    interactive=True,
                )
                new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入复刻的音色名称.", value='')
                new_name_path = gr.Textbox(label="保存后新的音色路径", lines=1, placeholder="复刻音色保存路径.", value='', interactive=False)
                prompt_asr_button.click(
                    fn = get_asr_result, 
                    inputs=[prompt_wav_upload], 
                    outputs=[prompt_msg]
                )
            with gr.Column():
                outputs_audio = gr.Audio(label="输出音频", interactive=True)
                with gr.Row():
                    submit_button = gr.Button("生成复刻音频", variant='primary')
                    save_button = gr.Button("保存刚刚推理的zero-shot音色")
            submit_button.click(
                fn = zeroshot,
                inputs=[tts_text, use_zero_shot, sft_dropdown, prompt_msg, prompt_wav_upload, new_name, new_dropdown],
                outputs=[outputs_audio]
            )
            save_button.click(
                fn = save_name,
                inputs=[new_name],
                outputs=[new_name_path],
            )
            refresh_button.click(
                fn = get_spk_new,
                inputs=[],
                outputs= new_dropdown,
            )
            
            
        # 页面6：管理页面
        with gr.Tab("管理页面") as tab5:
            with gr.Row():
                key = gr.Textbox(label="加密密钥", value="123456", interactive=True)
            with gr.Accordion("对话模型配置管理", open=False) as accordion:
                with gr.Group():
                    with gr.Row():
                        openai_api_input = gr.Textbox(label="OpenAI API地址", value=config.get("openai", "api"))
                        openai_api_key_input = gr.Textbox(
                            label="OpenAI API密钥", 
                            value=textarea_data_change(config.get("openai", "api_key")), 
                            lines=3
                        )
                    with gr.Row():
                        chatgpt_model_dropdown = gr.Dropdown(
                            choices=[
                                "qwen2:1.5b",
                                "qwen2:7b",
                                "llama3"
                            ], 
                            label="模型",
                            value=config.get("chatgpt", "model"), 
                        )
                    with gr.Row(): 
                        chatgpt_temperature_input = gr.Textbox(value=config.get("chatgpt", "temperature"), label="temperature")
                        chatgpt_max_tokens_input = gr.Textbox(value=config.get("chatgpt", "max_tokens"), label="max_tokens")
                        chatgpt_top_p_input = gr.Textbox(value=config.get("chatgpt", "top_p"), label="top_p")
            with gr.Accordion("语音合成配置管理", open=False) as accordion:
                with gr.Group():
                    with gr.Row():
                        tts_api = gr.Textbox(value=config.get("tts", "api"), label="TTS API地址")
                        tts_model_dropdown = gr.Dropdown(
                            choices=[
                                "cosyvoice",
                                "edgetts",
                            ], 
                            label="模型",
                            value='cosyvoice', 
                        )
                        edgetts_voice_dropdown = gr.Dropdown(
                            choices=[
                                'zh-CN-XiaoxiaoNeural',
                                'zh-CN-XiaoyiNeural',
                                'zh-CN-YunjianNeural',
                                'zh-CN-YunxiNeural',
                                'zh-CN-YunxiaNeural',
                                'zh-CN-YunyangNeural',
                                'zh-CN-liaoning-XiaobeiNeural',
                                'zh-CN-shaanxi-XiaoniNeural',],
                            label="edgetts角色",
                            value= "zh-CN-XiaoxiaoNeural")
                    
            with gr.Accordion("ASR配置管理", open=False) as accordion:
                with gr.Group():
                    with gr.Row():
                        asr_api = gr.Textbox(value=config.get("asr", "api"), label="ASR API地址")
            with gr.Group():
                with gr.Row():
                    gradio_save_local_checkbox = gr.Checkbox(value=config.get("gradio", "save_local"), label="保存配置到本地文件")
                with gr.Row():
                    save_btn = gr.Button("保存")
                    output_label = gr.Label(label="结果")

                    save_btn.click(
                        save_config,
                        inputs=[openai_api_input, openai_api_key_input,
                            chatgpt_model_dropdown, chatgpt_temperature_input, chatgpt_max_tokens_input, chatgpt_top_p_input,
                            tts_api, tts_model_dropdown, edgetts_voice_dropdown,
                            gradio_save_local_checkbox],
                        outputs=output_label,
                        js=None
                    )
           
         
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch()
 

      
            

if __name__ == "__main__":
    main()