import streamlit as st
import os

# Import functions from local modules
from src.scrapper import scrape_website
from src.rag_builder import get_vectorstore, get_rag_chain
from src.agent import handle_chat_input

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
                from langchain_text_splitters import RecursiveCharacterTextSplitter # Import here to avoid circular dependency
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
        handle_chat_input(prompt, st.session_state.rag_chain)
