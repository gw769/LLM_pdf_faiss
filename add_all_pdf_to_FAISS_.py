
import os

from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 全局变量 vector_db
vector_db = None


def find_pdfs_recursively(folder_path):
    """递归查找目录及其子目录中的所有 PDF 文件，忽略大小写"""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_pdf(pdf_file, embeddings):
    """处理单个 PDF 文件"""
    global vector_db
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    try:
        chunks = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50))
        print(f"Found {len(chunks)} chunks in {pdf_file}")
        added_db = FAISS.from_documents(chunks, embeddings)
        if vector_db is None:
            vector_db = added_db
        else:
            vector_db.merge_from(added_db)
        # 保存向量数据库
        if vector_db:
            vector_db.save_local('LLM_english.faiss')
            print('FAISS database saved!')
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

def main():
    global vector_db
    folder_path = 'announcements'
    pdf_files = find_pdfs_recursively(folder_path)
    #embeddings = HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-Qwen2-7B-instruct', model_kwargs={'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name="gtr-t5-large")

    #embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')

    # 尝试加载本地保存的向量数据库
    try:
        vector_db = FAISS.load_local('LLM_english.faiss', embeddings, allow_dangerous_deserialization=True)
        print('Loaded existing FAISS database.')
    except Exception as e:
        print('No existing FAISS database found, creating a new one.')
        vector_db = None

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")
        process_pdf(pdf_file, embeddings)

    

if __name__ == '__main__':
    main()