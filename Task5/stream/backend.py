import json
from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.llm import LLMChain
# from langchain_community.llms import Ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# 导入 dotenv 库，用于从 .env 文件加载环境变量，管理敏感数据，如 API 密钥。
from dotenv import load_dotenv
import os
import sys
import asyncio
import base64
# 导入本地加密库
from utils.encrypt import Encrypter



current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(current_path)
sys.path.append(parent_path)
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1/"

## 模拟一个密钥
SECRET_KEY = "51a6e85bd689409f"
encrypter = Encrypter(key=SECRET_KEY)

app = FastAPI()

load_dotenv()


# 历史对话文件路径
HISTORY_DIR = os.path.join(current_path, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
# 确保历史对话目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 如果历史对话文件不存在，则创建一个新的文件
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=4)

llm = ChatOpenAI(
            model="qwen2",
            openai_api_key="ollama",
            openai_api_base='http://localhost:11434/v1/',
            stop=['<|im_end|>'],
            streaming=True,
            max_tokens=256,
            temperature=0.9,
            top_p = 3
        )  


def load_history():
    """加载历史对话记录"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding='utf-8') as f:
            history = json.load(f)
            # 只保留最近的10轮对话
            return history[-10:]
    else:
        history = []
    return history

def save_history(history):
    """保存历史对话记录到文件"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)

# 创建聊天提示模板，包含系统消息、聊天历史占位符和人类消息模板。
def create_prompt(prompt_new='你是移动鸿鹄训练营第三小组设计的数字人助手.'):
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(prompt_new),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    return prompt
# 创建会话记忆，用于存储和返回会话中的消息。
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512)
# 创建一个 LLMChain 实例，包括语言模型、提示、详细模式和会话记忆。
prompt = create_prompt()
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)


class ChatRequest(BaseModel):
    key: str
    prompt: str = "You are a helpful assistant."
    query: str
    model: str = "qwen2",
    temperature: float=0.7,
    max_tokens: int=512,
    top_p: float=1,

@app.post("/chat_stream/")
async def chat_stream(messages: ChatRequest):
    prompt_in = create_prompt(messages.prompt)
    query = encrypter.receive_and_decrypt_message(messages.key, messages.query)
    llm = ChatOpenAI(
            model=messages.model,
            openai_api_key="ollama",
            openai_api_base='http://localhost:11434/v1/',
            stop=['<|im_end|>'],
            streaming=True,
            max_tokens=messages.max_tokens,
            temperature=messages.temperature,
            top_p = messages.top_p
        )
    llm_chain = prompt_in | llm
    # conversation = LLMChain(
    # llm=llm,
    # prompt=prompt_in,
    # verbose=True,
    # memory=memory
    # )
    # ret = conversation.stream({"question": query, "chat_history":memory.chat_history})
    # ret = llm_chain.stream({"question": query, "chat_history":memory.history})
    ret = llm_chain.stream({"question": query, "chat_history": list(memory.buffer)})
    def predict():
        text = ""
        for token in ret:
            # token = token['text']
            token = token.content
            js_data = {"code": "200", "msg": "ok", "data": token}
            yield f"data: {json.dumps(js_data,ensure_ascii=False)}\n\n"
            text += token
        print(f"--------------------------------------------\n{text}")
        save_history(memory.buffer)
        
    generate = predict()
    return StreamingResponse(generate, media_type="text/event-stream")

# ASR服务
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# asr model
# 修改为绝对路径
model_dir = os.path.join(parent_path, "ASR")
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)
class ASRItem(BaseModel):
    wav: str  # 输入音频

@app.post("/asr/")
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

# 文本instruct生成
from langchain.prompts import ChatMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate,SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser 

## 1.定义不同的子链的prompt模板
## 文本情感提取器
instruct_template = """
你是一个情感专家，你需要根据用户输入文本，分析文本中所蕴含的情感，以及朗读该文本时所需要的情感词汇

下面是需要你来回答的问题：
{input}

"""
## 文本情感标签生成器
emotion_template = """
你是一个语言专家，你需要根据用户输入，分析朗读文本所需的情绪、语气，并在文本中对应位置加入：<laughter>/</laughter>/<strong>/</strong>/[breath]/[laughter]六类标签，
给出加入标签后的文本。请注意，不能加入除了标签之外的内容，如果无法判断加入哪些标签，则返回原输入。

下面是需要你来回答的问题：
{input}

"""


# 2.创建prompt
instructor_template = ChatPromptTemplate.from_messages(
    [
        ("system", instruct_template),
        ("human", "我们走的每一步，都是我们策略的一部分；你看到的所有一切，包括我此刻与你交谈，所做的一切，所说的每一句话，都有深远的含义。"),
        ("ai", "A speaker with normal pitch, slow speaking rate, and sad emotion."),
        ("human", "深夜独行于荒芜的小巷，忽闻身后传来诡异的脚步声，我寒毛直竖，心跳如雷，无法抑制对未知危险的深深恐惧。"),
        ("ai", "A speaker with low pitch, slow speaking rate, and fearful emotion."),
        ("human", "{input}"),
    ]
)

emotioner_template = ChatPromptTemplate.from_messages(
    [
        ("system", instruct_template),
        ("human", "有时候，最简单的事情能让我们笑得最开心，就像是无意中听到的一个傻笑话"),
        ("ai", "[laughter]有时候，最简单的事情[laughter]能让我们笑得最开心，就像是无意中听到的一个傻笑话[laughter]"),
        ("human", "成功并不是预先设定的终点，它需要你一步一步地努力,持续地努力，最终将梦想变成现实。"),
        ("ai", "成功并不是预先设定的终点，它需要你一步一步地<strong>努力</strong>，持续地<strong>努力</strong>，最终将梦想变成现实。"),
        ("human", "{input}"),
    ]
)

# 创建一个通用的LLM链函数以避免重复代码
def create_llm_chain(template, llm):
    output_parser = StrOutputParser()
    return template | llm | output_parser

class InstructItem(BaseModel):
    query: str
    model: str = "qwen2"
    temperature: float=1
    max_tokens: int=1024
    top_p: float=1

@app.post("/instruct/")
def text_instruct_generator(item:InstructItem):
    # 先推理instruct,再预测emotion
    query = item.query
    llm = ChatOpenAI(
            model=item.model,
            openai_api_key="ollama",
            openai_api_base='http://localhost:11434/v1/',
            stop=['<|im_end|>'],
            streaming=True,
            max_tokens=item.max_tokens,
            temperature=item.temperature,
        )
    try:
        # 创建LLM链
        llm_chain_instruct = create_llm_chain(instructor_template, llm)
        inst = llm_chain_instruct.invoke({"input": query})
        
        llm_chain_emotion = create_llm_chain(emotioner_template, llm)
        emo = llm_chain_emotion.invoke({"input": query})
        
        return (inst, emo)
    except Exception as e:
        # 添加异常处理
        print(f"Error occurred: {e}")
        return {"error": "An error occurred during processing."}
    
    
# RAG部分
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os 
import sys
import logging
import datetime
import re
import json

# rag项目路径
rag_path = os.path.join(current_path, "RAG")
# 记忆能力构建
rag_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)

class RAGItem(BaseModel):
    query: str
    faiss_path: str=os.path.join(rag_path, 'vectorbase/LLM.faiss')
    embedModelName: str=os.path.join(rag_path, 'indexer_model\zpoint_large_embedding_zh')


@app.post("/RAG/")
def rag_chatbot(item:RAGItem):
    global rag_path
    embeddings = HuggingFaceEmbeddings(model_name = item.embedModelName)
    # 加载faiss向量库，用于知识召回
    vector_db=FAISS.load_local(item.faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever=vector_db.as_retriever(search_kwargs={"k":5})
    chat = llm
        
    # Prompt模板
    system_prompt=SystemMessagePromptTemplate.from_template(
            '你是一个对接问题排查机器人。你的任务是根据下述给定的已知信息回答用户问题。确保你的回复完全依据下述已知信息。不要编造答案。请用中文回答用户问题')
    user_prompt=HumanMessagePromptTemplate.from_template('''
    Answer the question based only on the following context:

    {context}

    Question: {query}
    ''')
    full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="rag_memory"),user_prompt])
    # full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,user_prompt])

    '''
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    ...
    <|im_start|>user
    Answer the question based only on the following context:

    {context}

    Question: {query}
    <|im_end|>
    <|im_start|>assitant
    ......
    <|im_end|>
    '''
    # Chat chain
    chat_chain={
            "context": itemgetter("query") | retriever,
            "query": itemgetter("query"),
            "rag_memory":itemgetter("rag_memory"),
        }|full_chat_prompt|chat
    # chat_chain={
    #     "context": itemgetter("query") | retriever,
    #     "query": itemgetter("query"),
    # }|full_chat_prompt|chat

    query = item.query
    response=chat_chain.invoke({'query':query, 'rag_memory':list(rag_memory.buffer)})
    res = response.content
    result_dict = {"code": 0, "msg": "ok", "res": res}
    
    return result_dict



 


# multi-prompt对话
from utils.Router_chain import RouterChainManager

class RouterChainItem(BaseModel):
    query: str
    doctor: bool=False
    secretary: bool=False
    english_teacher: bool=False
    custom_agents: bool=False

@app.post("/router/")
def router_chain(item:RouterChainItem):
    manager = RouterChainManager()
    chain = manager.create_router_chain(llm=llm,
                                        doctor=item.doctor, 
                                        secretary=item.secretary,
                                        english_teacher=item.english_teacher,
                                        custom_agents=item.custom_agents)
    response = chain.invoke({"input": item.query})
    res = response["text"]
    result_dict = {"code": 0, "msg": "ok", "res": res}
    
    return result_dict
        
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="127.0.0.1", port=10000)
    
    


