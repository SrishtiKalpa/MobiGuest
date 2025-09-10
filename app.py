import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage # Import for chat history

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_scraped" not in st.session_state:
    st.session_state.is_scraped = False
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "scraped_text" not in st.session_state:
    st.session_state.scraped_text = ""
if "visited_urls" not in st.session_state:
    st.session_state.visited_urls = set()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Web Scraping Function ---
def scrape_website(url, max_depth, current_depth=0):
    if current_depth > max_depth or url in st.session_state.visited_urls:
        return []

    st.session_state.visited_urls.add(url)
    scraped_content = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
        if body_text:
            scraped_content.append(body_text)

        if current_depth < max_depth:
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                if urlparse(absolute_url).netloc == urlparse(url).netloc:
                    scraped_content.extend(scrape_website(absolute_url, max_depth, current_depth + 1))

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
    except Exception as e:
        st.warning(f"An error occurred while processing {url}: {e}")

    return scraped_content

# --- RAG Pipeline Functions ---

def get_vectorstore(text_chunks):
    persist_directory = "./chroma_db"
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
    llm = ChatOllama(model="llama3.2") # Updated model to llama3.2

    # Contextualize question chain
    contextualize_q_system_prompt = """Given a chat history and the latest user question \n    which might reference context in the chat history, formulate a standalone question \n    which can be understood without the chat history. Do NOT answer the question, \n    just rephrase it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    # QA chain with persona
    qa_system_prompt = """You are a virtual front desk assistant for a hotel. Your name is HotelBot. \n    Answer user questions about the hotel based on the provided context. \n    If the answer is not in the context, politely state that you cannot assist with that specific query. \n    Use three sentences maximum and keep the answer concise.\n\n    {context}"""
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

# --- Main UI Layout ---
st.set_page_config(page_title="Virtual Front Desk Assistant")

st.title("Virtual Front Desk Assistant")
st.subheader("Enter a hotel website to get started")

url_input = st.text_input("Enter a website URL", key="url_input")

if st.button("Scrape and Initialize RAG"):
    if url_input:
        st.session_state.visited_urls = set()
        with st.spinner("Scraping..."):
            all_scraped_texts = scrape_website(url_input, max_depth=3)
            st.session_state.scraped_text = "\n\n".join(all_scraped_texts)
            st.session_state.is_scraped = True
            st.success("Scraping complete! Initializing RAG model...")

        if st.session_state.is_scraped and st.session_state.scraped_text:
            with st.spinner("Processing text and building RAG pipeline..."):
                # Text Processing
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.create_documents([st.session_state.scraped_text])

                # Embeddings and Vector Store
                st.session_state.vectorstore = get_vectorstore(texts)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever()

                # Retrieval Chain
                st.session_state.rag_chain = get_rag_chain(st.session_state.retriever)
                st.session_state.is_ready = True
                st.success("RAG model initialized and ready!")
        else:
            st.error("No content scraped to initialize RAG model.")
    else:
        st.warning("Please enter a URL.")

# --- Chat Interface ---
if st.session_state.is_ready:
    st.write("---") # Separator
    st.subheader("Ask HotelBot a question!")

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for user questions
    if prompt := st.chat_input("Your question:"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Prepare chat history for the RAG chain
            # Convert Streamlit chat history to Langchain's HumanMessage/AIMessage format
            langchain_chat_history = []
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    langchain_chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_chat_history.append(AIMessage(content=msg["content"]))

            # Stream response from RAG chain
            response_generator = st.session_state.rag_chain.stream(
                {"input": prompt, "chat_history": langchain_chat_history}
            )
            full_response = st.write_stream(response_generator)
            
            # Append assistant's full response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})