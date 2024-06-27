# LLM_pdf_faiss
use streamlit (frontend) and fast_api (backend) to show the easy use for langchain RAG
the llm use Ollama

<img width="1194" alt="image" src="https://github.com/gw769/LLM_pdf_faiss/assets/61015990/66515c07-923c-4852-88c5-a2f1b4b68b3f">


### Features
- **Streamlit Frontend**: Web interface powered by Streamlit for intuitive interaction.
- **FastAPI Backend**: Backend API built with FastAPI for robust and fast communication.
- **Faiss Integration**: Utilizes Faiss for efficient vector database querying.
- **PDF Parsing and Indexing**: Includes scripts for recursively finding and adding PDFs to the vector database.
- **Large Language Model Interaction**: use Ollama,so you should download ollama at first.

### Usage
To run the project locally, follow these steps:

1. **Backend (FastAPI)**:(one in cmd/powershell)
   ```bash
   cd /d D:\english-problem-qwen
   conda activate faiss
   python -m uvicorn fast_api:app --reload
   ```

2. **Frontend (Streamlit)**:(one in anothor cmd/powershell)
   ```bash
   conda activate faiss
   cd /d D:\english-problem-qwen
   streamlit run d:/english-problem-qwen/streamlit_app_v2.py
   ```

3. **Conda Environment Setup**:(optional)
   ```bash
   cd /d D:\english-problem-qwen
   conda env create -f environment_backup.yml
   ```

### Main Files
- **environment_backup.yml**: Conda environment backup file.
- **LLM_english.faiss**: Vector database for English language models.
- **announcements/**: Directory containing recursively stored PDFs.
- **add_all_pdf_to_FAISS_.py**: Script to recursively find and add PDFs to the vector database.
- **ask_qwen.py**: Script for querying JSON files using a large language model.
- **fast_api.py**: Backend API utilizing HTTP protocols.
- **streamlit.py**: Frontend interface utilizing HTTP protocols.


### ussage

1.run add_all_pdf_to_FAISS_.py, recursively search for all PDFs in the announcements directory and add them to the vector database. You can choose image text recognition OCR, LLM_english.faiss file is vector database.

2.ask_qwen.py, for testing, mainly open all json files, and use the big model to look at the answer and save it to the json file.

3.fast_api.py, backend api, connected with http protocol.

4.streamlit.py, front-end interface, connected with http protocol.

### Contribution
The next version i want to add three fucntion:
1. use the frontend to add pdf
2. try use docker
3. try add neo4j into this pipeline

### License
This project is no need licensed,so you can do whatever you want to use this code.

