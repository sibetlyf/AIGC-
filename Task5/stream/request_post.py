import json
import logging
from copy import deepcopy
import requests
import base64
import asyncio
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
# local
from utils.common import Common
from utils.logger import Configure_logger
# 导入敏感词检测系统
from utils.SensitivePy.sensetive_detector import KeywordFilter
# 导入本地加密库
from utils.encrypt import Encrypter

url = "http://127.0.0.1:10000"



## 模拟一个密钥
SECRET_KEY = "51a6e85bd689409f"
encrypter = Encrypter(key=SECRET_KEY)
# 初始化敏感词过滤器
keyword_filter = KeywordFilter()
# 从文件中加载敏感词
keyword_filter.load_keywords_from_file()

class ChatGPT():
    # 设置会话初始值
    # session_config = {'msg': [{"role": "system", "content": config_data['chatgpt']['preset']}]}
    session_config = {}
    sessions = {}
    current_key_index = 0
    data_openai = {}
    data_chatgpt = {}
    def __init__(self, api_key, api, model='qwen2', temperature=0.7, max_tokens=1024, top_p=1, prompt='你是移动鸿鹄训练营第3小组设计的数字人助手'):
        self.common = Common()
        self.api_key = api_key
        self.api = api
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.prompt = prompt
        
        # 设置会话初始值
        self.session_config = {'msg': [{"role": "system", "content": prompt}]}

    def get_chat_session(self, sessionid='user'):
        """
        获取指定 ID 的会话，如果不存在则创建一个新的会话
        :param sessionid: 会话 ID
        :return: 指定 ID 的会话
        """
        sessionid = str(sessionid)
        if sessionid not in self.sessions:
            config = deepcopy(self.session_config)
            config['id'] = sessionid
            config['msg'].append({"role": "system", "content": "current time is:" + self.common.get_bj_time()})
            self.sessions[sessionid] = config
        return self.sessions[sessionid]
    
    def reset_configs(self, config_data):
        self.api_key = config_data['openai']['api_key']
        self.api = config_data['openai']['api']
        self.model = config_data['chatgpt']['model']
        self.temperature = config_data['chatgpt']['temperature']
        self.max_tokens = config_data['chatgpt']['max_tokens']
        self.top_p = config_data['chatgpt']['top_p']
        
    
    def chat(self, prompt_new=None, messages=None, sessionid='user'):
        """
        ChatGPT 对话函数
        :prompt_new: 用户输入的prompt
        :messages: 当前会话内容
        :return: ChatGPT 返回的回复内容
        """
        if prompt_new is not None:
            self.prompt = prompt_new
        if messages is None:
            raise ValueError('messages is None')
        
        # 过滤敏感词
        prompt_new, messages = keyword_filter.filter_text(prompt_new), keyword_filter.filter_text(messages)
        print("prompt:", prompt_new, "\nmessages:", messages)
        # 如果有敏感词，则直接返回
        if "<sensetive word!>" in prompt_new or "<sensetive word!>" in messages:
            
           yield "检测到敏感词，请修改您的输入！"
        else:
            # 获取当前会话
            session = self.get_chat_session(sessionid)

            # 将用户输入的消息添加到会话中
            session['msg'].append({"role": "user", "content": messages})

            # 添加当前时间到会话中
            session['msg'][1] = {"role": "system", "content": "current time is:" + self.common.get_bj_time()}
            
            # 设置请求头
            headers = {'Accept': 'text/event-stream'}
            # 发送POST请求地址
            url = self.api + '/chat_stream/'
            # 加密数据
            encrypted_aes_key, encrypted_message = encrypter.send_encrypted_message(messages)
            # 构建请求数据
            data = {
                "key": encrypted_aes_key,
                "prompt": self.prompt,
                "query": encrypted_message,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
            
            response = requests.post(
                        url=url,
                        headers=headers,
                        json=data,
                        stream=True,
                    )
            resp = ""
            try:
                # 处理流式响应
                # for line in response.iter_lines(decode_unicode=True):
                for line in response:
                    if line:
                        # 去除可能的前导空白
                        line = line.strip()
                        # 忽略空行
                        if not line:
                            continue
                        
                        data = line[5:].strip()
                        # 解析JSON数据
                        try:
                            json_data = json.loads(data)
                            # 输出模型返回的内容
                            # print(json_data['data'])
                            token = json_data['data']
                            resp += token
                            # print(resp)
                            yield token
                        except ValueError:
                            print(f"Invalid JSON format: {data}")
            except KeyboardInterrupt:
                pass
            finally:
                # 将 ChatGPT 返回的回复消息添加到会话中
                session['msg'].append({"role": "assistant", "content": resp})
                # 输出会话 ID 和 ChatGPT 返回的回复消息
                logging.info("会话ID: " + str(sessionid))
                logging.debug("ChatGPT返回内容: ")
                logging.debug(messages)
                
                response.close()


# ASR请求

def asr_api_client(wav_path):
    global url
    headers = {'Content-Type': 'application/json'}
    try:
        with open(wav_path, "rb") as f:
            wav = base64.b64encode(f.read()).decode()
        
        data = {"wav": wav}
        response = requests.post(url + "/asr", headers=headers, json=data)
        response.raise_for_status()  # 检查HTTP状态码是否正常
        
        response_data = response.json()
        
        if response_data['code'] == 0:
            return response_data['res']
        else:
            return response_data['msg']
    
    except requests.RequestException as e:
        # 处理网络请求异常
        return f"Request failed: {e}"
    except Exception as e:
        # 其他异常处理
        return f"An error occurred: {e}"


# instruct生成请求
import re
from utils.coze import Coze
def split_response_by_input_prefix(text, response):
    # 获取输入文本的前三个字符
    prefix = text[:3]
    
    # 使用正则表达式查找与前三个字符相匹配的第一个位置
    match = re.search(re.escape(prefix), response)
    
    # 如果找到了匹配
    if match:
        start_index = match.start()
        
        # 切分输出文本
        instruct = response[:start_index].strip()
        emo = response[start_index:].strip()
        
        return instruct, emo
    else:
        return None, None

os.environ['COZE_API_TOKEN'] = 'pat_4AyvTNR58wBDZIm3NyHZNCG8k4W7y36lRsbcvJnQRkJ5W06FcurX8cQmL0lgwBhW'
os.environ['COZE_BOT_ID'] = "7407371111056703524"
def get_instrut(text:str,):
    chat = Coze(api_token=os.environ['COZE_API_TOKEN'],
                bot_id=os.environ['COZE_BOT_ID'],
                max_chat_rounds=20,
                stream=True)
    try:
        response = chat.chat(text)
        instruct, emo = split_response_by_input_prefix(text, response)
        print('instruct:', instruct, "\n--------\n", 'emotion:', f'{emo}\n')
        print(response)
        return instruct, emo
    except Exception as e:
        print(e)
        return "Sorry, I don't know how to answer that."

# RAG请求
def get_rag_response(faiss_path, messages):
    # 验证路径有效性
    if not os.path.exists(faiss_path):
        print("Error: The faiss file does not exist!")
        return False

    try:
        # 设置请求头
        headers = {'Accept': 'text/event-stream'}
        # 发送POST请求
        full_url = url + '/RAG/'
        # 构建请求数据
        data = {
            "query": messages,
            "faiss_path": faiss_path,
        }

        # 使用with语句确保流式响应正确关闭
        response = requests.post(url=full_url, headers=headers, json=data, stream=True,)
        response.raise_for_status()  # 检查HTTP状态码是否正常
        response_data = response.json()
        if response_data['code'] == 0:
            return response_data['res']
        else:
            return response_data['msg']
    except requests.exceptions.RequestException as e:
        # 提供具体的错误信息
        print(f"Request failed: {e}")
        raise

    except Exception as e:
        # 其他未知错误
        print(f"An unexpected error occurred: {e}")
        raise
    
# Router Chain请求
def get_router_response(query, doctor=False, secretary=False, english_teacher=False, custom_agents=False):
    try:
        # 设置请求头
        headers = {'Accept': 'text/event-stream'}
        # 发送POST请求
        full_url = url + '/router/'
        # 构建请求数据
        data = {
            "query": query,
            "doctor": doctor,
            "secretary": secretary,
            "english_teacher": english_teacher,
            "custom_agents": custom_agents
        }

        # 使用with语句确保流式响应正确关闭
        response = requests.post(url=full_url, headers=headers, json=data, stream=True,)
        response.raise_for_status()  # 检查HTTP状态码是否正常
        response_data = response.json()
        if response_data['code'] == 0:
            return response_data['res']
        else:
            return response_data['msg']
    except requests.exceptions.RequestException as e:
        # 提供具体的错误信息
        print(f"Request failed: {e}")
        raise

    except Exception as e:
        # 其他未知错误
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    # instruct测试
    # text = "我是鸿鹄梧桐训练营第三组设计的数字人，提供舒适自然的语音合成能力"
    # instruct, emo = get_instrut(text)
    # print(instruct, '\n', emo)
    
    # RAG测试
    # from RAG.indexer import indexe_pdf
    # pdf_path = r'E:\pythonProject\AIGC-\Task5-LLM-chat\stream\RAG\aigc.pdf'
    # indexe_pdf(pdf_path=pdf_path)
    # text = "请总结全文的主旨"
    # faiss_path = r'E:\pythonProject\AIGC-\Task5-LLM-chat\stream\RAG\vectorbase\LLM.faiss'
    # print(get_rag_response(faiss_path=faiss_path, messages=text))
    
    # Router Chain测试
    text = "我眼睛有点痛，怎么办"
    print(get_router_response(text, True, True, True, False))
            
        
        
        