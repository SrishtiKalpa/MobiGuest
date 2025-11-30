import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

def get_vectorstore(text_chunks, agent_name):
    # Sanitize the agent name for use in file paths
    import re
    safe_agent_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(agent_name))
    
    # Point to the backend's data directory with agent-specific subdirectory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../engine'))
    data_dir = os.path.join(backend_dir, 'data')
    persist_directory = os.path.join(data_dir, 'chroma_db', safe_agent_name)
    
    try:
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Check if the directory exists and has content
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            st.info("Loading vector store from disk...")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            st.success(f"Vector store for agent '{agent_name}' loaded from: {persist_directory}")
        else:
            st.info("Creating new vector store...")
            vectorstore = Chroma.from_documents(
                text_chunks, 
                embeddings, 
                persist_directory=persist_directory
            )
            # Explicitly persist the vector store
            vectorstore.persist()
            st.success(f"New vector store for agent '{agent_name}' created at: {persist_directory}")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        st.error(f"Attempted to use directory: {persist_directory}")
        return None

def get_rag_chain(retriever, agent_name=""):
    llm = ChatOllama(model="llama3.2")

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just rephrase it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = f"""You are a virtual assistant. Your name is {agent_name if agent_name else 'Assistant'}. \
    Answer user questions about the hotel based on the provided context. \
    If the answer is not in the context, politely state that you cannot assist with that specific query. \
    Use three sentences maximum and keep the answer concise. \

    {{context}}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
