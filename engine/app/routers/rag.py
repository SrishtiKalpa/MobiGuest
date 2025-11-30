from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import os

router = APIRouter()

# Models
class RAGRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class RAGResponse(BaseModel):
    response: str
    sources: List[str] = []

# Initialize components
embeddings = OllamaEmbeddings()
vectorstore = None
retriever = None

@router.post("/query", response_model=RAGResponse)
async def query_rag(rag_request: RAGRequest):
    """
    Query the RAG system with a question and optional context.
    """
    try:
        if not vectorstore or not retriever:
            raise HTTPException(status_code=400, detail="RAG system not initialized. Please load documents first.")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(rag_request.query)
        
        # Format sources
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        # Generate response (placeholder - implement actual RAG logic)
        response = f"Response to: {rag_request.query}\n\n"
        response += "Relevant context:\n"
        for i, doc in enumerate(docs, 1):
            response += f"{i}. {doc.page_content[:200]}...\n"
        
        return RAGResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load_documents")
async def load_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load documents into the vector store.
    """
    global vectorstore, retriever
    
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            documents,
            embedding=embeddings,
            metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever()
        
        return {"status": "success", "message": f"Loaded {len(documents)} documents into vector store"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
