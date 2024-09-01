from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)



def indexe_pdf(pdf_path='test.pdf'):
    # 解析pdf，切成chunk片段
    loader = PyPDFLoader(pdf_path, extract_images=True) # 使用OCR解析pdf中图片里面的文字
    chunks=loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10))
    
    # 加载embedding模型，用于将chunk向量化
    embedModelName = os.path.join(current_path, 'indexer_model\zpoint_large_embedding_zh')
    embedding = HuggingFaceEmbeddings(model_name = embedModelName)
    
    # 将chunk插入到faiss本地向量数据库
    vector_db=FAISS.from_documents(chunks, embedding)
    vector_db_save_path = os.path.join(current_path, 'vectorbase', 'LLM.faiss')
    vector_db.save_local(vector_db_save_path)

    print('faiss saved!')
    return vector_db_save_path

# 加载
def load_vector_store():
    return FAISS.load_local(os.path.join(current_path, 'vectorBase', 'LLM.faiss'), HuggingFaceEmbeddings())

if __name__ == '__main__':
    indexe_pdf()
