#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1. Document loader and vectorstore
def load_docs(pdf_paths):
    text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Step 2. Define base LLM
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation")

# Step 3. Create the tools (wrapped functions)
def refine_query_tool(query: str) -> str:
    """
    Acts like the Context Agent.
    Reformulates vague queries into more precise ones.
    """
    reformulation_prompt = f"Make this query clear, self-contained, and specific:\n\n{query}"
    return llm(reformulation_prompt)

def retrieve_answer_tool(query: str) -> str:
    """
    Acts like the Content Agent.
    Retrieves and answers using the RAG system.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return rag_chain.run(query)

# Step 4. Register tools
tools = [
    Tool(
        name="Context Understanding Tool",
        func=refine_query_tool,
        description="Use this when the query seems unclear or vague."
    ),
    Tool(
        name="RAG Content Tool",
        func=retrieve_answer_tool,
        description="Use this to retrieve and answer based on document knowledge."
    ),
]

# Step 5. Initialize an Agent that decides between tools
agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Example run ---
#if __name__ == "__main__":
#    pdf_files = ["sample1.pdf", "sample2.pdf"]
#    text_data = load_docs(pdf_files)
#    vectorstore = create_vectorstore(text_data)
#
#    query = "What are the laws related to AI in Europe?"
#    response = agent.run(query)
#
#    print("\n🤖 Final Answer:\n", response)
#


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Agentic_RAG.ipynb')


# In[ ]:




