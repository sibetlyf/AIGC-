from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os 
import sys
import logging
import datetime
import re
import json

"""
    RAG_Agent
    1. 读取日志文件，获取对话历史记录
    2. 构建prompt模板
    3. 构建chat chain
    4. 调用chat chain
    5. 将chat chain的输出添加到对话历史记录中
    6. 将对话历史记录保存到日志文件
    7. 循环
"""
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1/"

# 创建日志文件夹
log_floder = os.path.join(current_path, 'logs')
os.makedirs(log_floder, exist_ok=True)
# 设置日志文件
log_filename = f'history_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log'
log_file_path = os.path.join(log_floder, log_filename)
# 配置日志
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

chat_history = []
history = ChatMessageHistory()


def parse_log(log_file_path=log_file_path):
    """从给定日志文件中解析对话历史记录"""
    global history
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'\[(user|assistant)\] (.*)', line)
            if match:
                role, content = match.groups()
                if role == 'user':
                    history.add_user_message(content)
                elif role == 'assistant':
                    history.add_ai_message(content)
                    logging.info("Parsed assistant message: %s", content)
    logging.info("Finished parsing log file.")

def save_chat_history_to_json(history, filename):
    """将对话历史记录保存为 JSON 文件"""
    chat_history_data = [
        {'role': 'user', 'content': msg.content} if isinstance(msg, HumanMessage) else
        {'role': 'assistant', 'content': msg.content} for msg in history.messages
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chat_history_data, f, ensure_ascii=False, indent=4)
    logging.info("Saved chat history to JSON file: %s", filename)


def RAG_Agent(embedModelName=os.path.join(current_path, 'indexer_model\zpoint_large_embedding_zh'),
              faiss_path=os.path.join(current_path, 'vectorbase\LLM.faiss'),
              messages=None,
              ):

        embeddings = HuggingFaceEmbeddings(model_name = embedModelName)
        # 加载faiss向量库，用于知识召回
        vector_db=FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        retriever=vector_db.as_retriever(search_kwargs={"k":5})
        chat=ChatOpenAI(
            model="qwen2",
            openai_api_key="ollama",
            openai_api_base='http://localhost:11434/v1/',
            stop=['<|im_end|>']
        )  
        
        # Prompt模板
        system_prompt=SystemMessagePromptTemplate.from_template(
            '你是一个对接问题排查机器人。你的任务是根据下述给定的已知信息回答用户问题。确保你的回复完全依据下述已知信息。不要编造答案。请用中文回答用户问题')
        user_prompt=HumanMessagePromptTemplate.from_template('''
        Answer the question based only on the following context:

        {context}

        Question: {query}
        ''')
        full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])

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
                "chat_history":itemgetter("chat_history"),
            }|full_chat_prompt|chat

        global history
        global chat_history
        parse_log(log_file_path)
        query = messages
        response=chat_chain.invoke({'query':query,'chat_history':chat_history})
        chat_history.extend((HumanMessage(content=query),response))
        chat_history=chat_history[-10:] # 最新10轮对话
        history.add_user_message(query)
        history.add_ai_message(response.content)
        
        # 保存对话历史记录为 JSON 文件
        json_filename = f'chat_history_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json'
        json_file_path = os.path.join(log_floder, json_filename)
        save_chat_history_to_json(history, json_file_path)
    
        return response

def start_chat():
    global history
    while True:
        query = input('query:')
        if query == 'exit':
            break
        response=RAG_Agent(messages=query)
        print(response.content)
            
if __name__ == '__main__':
    start_chat()
