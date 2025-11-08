#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ----------- Streamlit UI --------------
from RAG_APP import get_pdf_text,get_text_chunks,get_vectorstore,get_conversation_chain
import streamlit as st
st.set_page_config(page_title="📚 Multi-PDF Chatbot", layout="wide")
st.title("📚 Chat with Multiple PDFs")

st.sidebar.header("Upload your PDF files")
pdf_docs = st.sidebar.file_uploader(
    "Upload multiple PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if pdf_docs:
    with st.spinner("📄 Reading and processing PDF files..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)
    st.success("✅ PDFs processed successfully! You can now start chatting.")

    # Chat input and response area
    if "conversation" not in st.session_state:
        st.session_state.conversation = conversation_chain
        st.session_state.chat_history = []

    user_query = st.text_input("Ask a question about the PDFs:")

    if user_query:
        with st.spinner("💬 Generating response..."):
            response = st.session_state.conversation({"question": user_query})
            answer = response["answer"]
            st.session_state.chat_history.append((user_query, answer))

        # Display chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")

else:
    st.info("👈 Upload one or more PDF files from the sidebar to get started.")


# In[7]:


#get_ipython().system('jupyter nbconvert --to script APP1.ipynb')


# In[ ]:




