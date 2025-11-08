#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install streamlit langchain chromadb pypdf sentence-transformers transformers accelerate


# In[1]:


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

# ----------- Helper Functions --------------

@st.cache_data(show_spinner=True)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.replace("\n", " ").strip() + " "
    return text

@st.cache_data(show_spinner=True)
def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource(show_spinner=True)
def get_vectorstore(text_chunks):
    """Convert chunks into embeddings and store in FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

@st.cache_resource(show_spinner=True)
def get_conversation_chain(_vectorstore):
    """Build the conversational retrieval chain."""
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    return conversation_chain



# In[13]:


#get_ipython().system('jupyter nbconvert --to script RAG_APP.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:




