1.运行
    前端为streamlit
    后端为fastapi
    需要开两个cmd/powershell分别运行

    cd /d D:\english-problem-qwen
    conda activate faiss
    python -m uvicorn fast_api:app --reload

    和

    conda activate faiss
    cd /d D:\english-problem-qwen
    streamlit run d:/english-problem-qwen/streamlit_app_v2.py

2.conda环境配置，environment_backup.yml conda环境备份文件
    cd /d D:\english-problem-qwen
    备份： conda env export > environment_backup.yml ，进去之后改name
    创建： conda env create -f environment_backup.yml


3.ussage

    1.run add_all_pdf_to_FAISS_.py，递归的寻找announcements目录下所有pdf并添加进向量数据库，可以选择图片文字识别OCR，    LLM_english.faiss 向量数据库
    2.ask_qwen.py，测试用，主要是打开json所有文件，并用大模型看一下答案并保存到json文件
    3.fast_api.py,后端api，用http协议链接
    4.streamlit.py，前端界面，用http协议链接
