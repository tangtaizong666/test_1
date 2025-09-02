from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
#嵌入模型
from langchain_openai import OpenAIEmbeddings
#向量数据库
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ai_agent(api_key,base_url,memory,file,question):
    models=ChatOpenAI(
        model='gpt-3.5-turbo',
        api_key=api_key,
        base_url=base_url)
    #获取文件中的二进制信息
    file_content=file.read()
    temp_path='temp.pdf'
    with open(temp_path,'wb') as f:
        f.write(file_content)

    #创建加载器实例
    loader=PyPDFLoader(temp_path)
    doc=loader.load()

    #创建分割器实例
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=['\n','。','?','!','，','`','']
    )
    texts=text_splitter.split_documents(doc)
    embeddings_model=OpenAIEmbeddings(model='text-embedding-3-large',
                            api_key=api_key,
                            base_url=base_url,
                            dimensions=1024)
    db=FAISS.from_documents(texts,embeddings_model)
    retriever=db.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(
        llm=models,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({'chat_history':memory,'question':question})
    return response

'''
def ai_agent(openai_api_key,base_url, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key,base_url=base_url)
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-large',
                                        api_key=openai_api_key,
                                        base_url=base_url,
                                        dimensions=1024)
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
'''