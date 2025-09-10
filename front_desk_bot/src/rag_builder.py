import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

def get_vectorstore(text_chunks):
    persist_directory = "./front_desk_bot/data/chroma_db" # Updated path
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        st.info("Loading vector store from disk...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        st.info("Creating new vector store...")
        vectorstore = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)
        st.success("Vector store created and saved.")
    return vectorstore

def get_rag_chain(retriever):
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

    qa_system_prompt = """You are a virtual front desk assistant for a hotel. Your name is HotelBot. \
    Answer user questions about the hotel based on the provided context. \
    If the answer is not in the context, politely state that you cannot assist with that specific query. \
    Use three sentences maximum and keep the answer concise. \

    {context}"""
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
