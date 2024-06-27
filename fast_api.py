import json
import os

from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

app = FastAPI()

selected_model = 'qwen:32b'
model=Ollama(model="qwen:32b")
embeddings = HuggingFaceEmbeddings(model_name="gtr-t5-large")
vector=FAISS.load_local('LLM_english.faiss',embeddings,allow_dangerous_deserialization=True)
retriever=vector.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.7}
)

template2 = """ You are a chatbot that provides information to students, collaborators and professors at the University of Wollongong.Your tone should be professional and informative. Keep your responses brief and only answer the question asked by the user.If you don't know the answer, state so. You have to answer in the language used by the user. Use the following context to response the questions.
Context: {context}
History: {history}

Question: {question}
Chatbot:
"""

prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template2,
    )

#注意documents_chain的prompt格式要求
#document_chain = create_stuff_documents_chain(model, prompt)
#注意retrieval_chain要求传入input,否则报错KeyError: 'input'，retrieval_docs = (lambda x: x["input"]) | retriever


memory = ConversationBufferMemory(
            memory_key="history", 
            return_messages=True,
            input_key="question" 
    )
# 创建LLMChain实例
retrieval_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory
            }
        )

class Query(BaseModel):
    query: str
    what_model_choice_LLM: str
    score_threshold: float
print('FastAPI server started!')

@app.post("/query/")
def get_response(query: Query):
    global model, selected_model, retrieval_chain,retriever
    if query.what_model_choice_LLM != selected_model or query.score_threshold!=0.7:
        if query.what_model_choice_LLM == 'qwen:32b':
            model = Ollama(model="qwen:32b")
        elif query.what_model_choice_LLM == 'qwen2:7b':
            model = Ollama(model="qwen2:7b")
        elif query.what_model_choice_LLM == 'qwen:14b':
            model = Ollama(model="qwen:14b")
        else:
            model = Ollama(model="llama3:8b")
        selected_model = query.what_model_choice_LLM
        memory = ConversationBufferMemory(
            memory_key="history", 
            return_messages=True,
            input_key="question" 
        )
        retriever=vector.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': query.score_threshold}
        )
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory
            }
        )

    response1 = retrieval_chain.invoke({"query": query.query})
    print(response1['result'])
    return {"response": response1['result']}

def find_pdfs_recursively(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files



def process_pdf(pdf_file, embeddings):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    try:
        chunks = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50))
        print(f"Found {len(chunks)} chunks in {pdf_file}")
        added_db = FAISS.from_documents(chunks, embeddings)
        if vector_db is None:
            vector_db = added_db
        else:
            vector_db.merge_from(added_db)
        if vector_db:
            vector_db.save_local('LLM_english.faiss')
            print('FAISS database saved!')
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

def load_dataset(json_path):
    all_data = []
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        id = item['id'].strip()
        question = item['original_text'].strip()
        ans = item['ans'].strip()
        all_data.append([id, question, ans])
    return all_data