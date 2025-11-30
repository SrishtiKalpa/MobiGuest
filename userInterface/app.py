import streamlit as st
import os
import tempfile
import logging

import PyPDF2
from docx import Document
import pandas as pd

# Import functions from local modules
from src.scrapper import scrape_website
from src.rag_builder import get_vectorstore, get_rag_chain
from src.agent import handle_chat_input

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Helpers for document processing ----------
def _process_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def _process_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _process_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_string()


def _process_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "visited_urls" not in st.session_state:
    st.session_state.visited_urls = set()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "agent_name" not in st.session_state:
    st.session_state.agent_name = ""

# --- Main UI Layout ---
st.set_page_config(page_title="Virtual Front Desk Assistant")

st.title("Virtual Front Desk Assistant")
st.caption("Build a hotel assistant from a website, documents, and custom instructions.")

col_left, col_right = st.columns(2)

with col_left:
    st.session_state.agent_name = st.text_input("Agent name", value=st.session_state.agent_name or "", help="Name your virtual assistant")
    url_input = st.text_input("Website URL (optional)", key="url_input")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF / DOCX / TXT / CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

with col_right:
    extra_context = st.text_area(
        "Extra context / instructions (optional)",
        placeholder="Describe the hotel, policies, FAQs, etc.",
        height=150,
    )

if st.button("Scrape / Upload and Initialize RAG", type="primary"):
    if not (url_input or uploaded_files or extra_context.strip()):
        st.warning("Please enter a URL, upload at least one document, or provide extra context.")
    else:
        combined_parts = []

        # Add extra text context first
        if extra_context.strip():
            combined_parts.append(extra_context.strip())

        # Scrape website if provided
        if url_input:
            st.session_state.visited_urls = set()
            with st.spinner("Scraping website..."):
                scraped_texts = scrape_website(url_input, max_depth=3)
                if scraped_texts:
                    combined_parts.append("\n\n".join(scraped_texts))
                    st.success("Website scraping complete.")
                else:
                    st.warning("No text content found on the provided website.")

        # Process uploaded documents
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                for uploaded in uploaded_files:
                    try:
                        suffix = os.path.splitext(uploaded.name)[1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = tmp.name

                        if suffix == ".pdf":
                            doc_text = _process_pdf(tmp_path)
                        elif suffix == ".docx":
                            doc_text = _process_docx(tmp_path)
                        elif suffix == ".csv":
                            doc_text = _process_csv(tmp_path)
                        else:
                            doc_text = _process_txt(tmp_path)

                        if doc_text.strip():
                            combined_parts.append(doc_text)
                            st.success(f"Processed: {uploaded.name}")
                        else:
                            st.warning(f"No text extracted from: {uploaded.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded.name}: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

        full_corpus = "\n\n".join(part for part in combined_parts if part.strip())

        if not full_corpus.strip():
            st.error("No usable text could be extracted from the provided sources.")
        else:
            with st.spinner("Building RAG pipeline..."):
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.create_documents([full_corpus])

                st.session_state.vectorstore = get_vectorstore(docs, st.session_state.agent_name)
                if st.session_state.vectorstore is None:
                    st.error("Failed to create vector store.")
                else:
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                    st.session_state.rag_chain = get_rag_chain(st.session_state.retriever, st.session_state.agent_name)
                    if st.session_state.rag_chain is None:
                        st.error("Failed to create RAG chain.")
                    else:
                        st.session_state.is_ready = True
                        st.success(f"RAG model for {st.session_state.agent_name} is ready!")

# --- Chat Interface ---
if st.session_state.is_ready:
    st.write("---")
    st.subheader(f"Ask {st.session_state.agent_name} a question!")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question:"):
        handle_chat_input(prompt, st.session_state.rag_chain)
