import json
import os

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 假设这是你的PDF文件列表
def load_dataset(json_path):

    all_data = []

    # Load data
    with open(json_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    
    # Initialize the tokenizer
    
    for item in data:
        id = item['id'].strip()
        question = item['original_text'].strip()
        ans = item['ans'].strip()
        all_data.append([id,question, ans])

    return all_data
def main():
    json_path = 'test.json'
    data = load_dataset(json_path)
    print(data[0])

    model=Ollama(model="qwen:14b")
    #reports_vector_db = Chroma(persist_directory=REPORTS_CHROMA_PATH, embedding_function=embedding_function)
    #embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base') 
    embeddings = HuggingFaceEmbeddings(model_name="gtr-t5-large")
    vector=FAISS.load_local('LLM_english.faiss',embeddings,allow_dangerous_deserialization=True)
    #retriever = vector.as_retriever()
    retriever=vector.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.7}
    )
    #retriever=vector.as_retriever()(search_kwargs={'filter': {'paper_title':"南方宝丰混合型证券投资基金2023年第3季度报告.PDF"}})
    template = """ 你是一位为 University of Wollongong的学生、合作者和教授提供信息的聊天机器人。
    你的语气应该专业且信息丰富。保持回答简洁，只回答用户提出的问题。
    如果不知道答案，直接说明。你必须中文回答问题。
        
    Context: {context}
    History: {history}

    Question: {question}
    Chatbot:
    """
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
    #retrieval_chain = create_retrieval_chain(retriever,model, prompt,memory)#response2
     
    qa_pairs = []
    for id,original_text,ans in data:
        response1 = retrieval_chain.invoke({"query": original_text})
        '''response1 是通过 llm_chain 模型生成的回答，该模型使用了 prompt作为输入模板,Faiss为retriever的向量数据库,同时memory能够记忆上下文。
        '''
        print('-'*50)
        print("问题id"+id+" \n原问题是:"+original_text)
        print("qwen的回答是："+response1['result'])
        print('-'*30)
        print("正确的答案right_answer应该是:"+ans)
        print('-'*50)
        qa_pair = {
                "id":id,
                "original_text":original_text,
                "ans":ans,
                "qwen2_answer": response1['result']
        }
        qa_pairs.append(qa_pair)
    with open("test_answer.json", "w", encoding="utf-8") as answer_file:
        json.dump(qa_pairs, answer_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()