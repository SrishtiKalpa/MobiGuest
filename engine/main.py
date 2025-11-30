from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import shutil
import time
from pathlib import Path
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="AI Agent API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

api_router = APIRouter(prefix="/api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AgentCreate(BaseModel):
    name: str
    context: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    agent_name: str
    message: str
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, str]]

# Global state (in production, use a proper database)
AGENTS_DIR = "./chroma_db"
Path(AGENTS_DIR).mkdir(exist_ok=True)

# Helper functions
def get_rag_chain(retriever, bot_name: str):
    logging.info(f"Initializing RAG chain for bot: {bot_name}")
    try:
        llm = ChatOllama(model="llama3.2")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        logging.info(f"RAG chain for {bot_name} successfully created.")
        return rag_chain
    except Exception as e:
        logging.error(f"Error creating RAG chain for bot {bot_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing RAG chain: {str(e)}")

# API Endpoints
@api_router.post("/agents", response_model=Dict)
async def create_agent(agent: AgentCreate):
    """Create a new agent with the given name and optional context."""
    logging.info(f"Received request to create agent: {agent.name}")
    # Use a normalized, lowercase name for filesystem and collection consistency
    safe_name = "".join(c if c.isalnum() else "_" for c in agent.name).lower()
    persist_dir = os.path.join(AGENTS_DIR, safe_name)
    
    try:
        if os.path.exists(persist_dir):
            logging.warning(f"Agent {agent.name} already exists. Path: {persist_dir}")
            raise HTTPException(status_code=400, detail=f"Agent '{agent.name}' already exists")
        
        os.makedirs(persist_dir, exist_ok=True)
        logging.info(f"Agent directory created for {agent.name} at {persist_dir}")
        
        return {"message": f"Agent '{agent.name}' created successfully", "agent_name": agent.name}
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logging.error(f"Error creating agent {agent.name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@api_router.get("/agents", response_model=List[Dict])
async def list_agents():
    """List all available agents."""
    logging.info("Received request to list agents.")
    agents = []
    try:
        if os.path.exists(AGENTS_DIR):
            for agent_dir in os.listdir(AGENTS_DIR):
                agent_path = os.path.join(AGENTS_DIR, agent_dir)
                if os.path.isdir(agent_path):
                    agents.append({
                        "name": agent_dir.replace("_", " ").title(),
                        "created_at": os.path.getctime(agent_path),
                        "updated_at": os.path.getmtime(agent_path)
                    })
        logging.info(f"Successfully listed {len(agents)} agents.")
        return {"agents": agents}
    except Exception as e:
        logging.error(f"Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@api_router.delete("/agents/{agent_name}", response_model=Dict)
async def delete_agent(agent_name: str):
    """Delete an agent and its associated data."""
    logging.info(f"Received request to delete agent: {agent_name}")
    safe_name = "".join(c if c.isalnum() else "_" for c in agent_name).lower()
    persist_dir = os.path.join(AGENTS_DIR, safe_name)

    try:
        if not os.path.exists(persist_dir):
            logging.warning(f"Attempted to delete non-existent agent {agent_name} at {persist_dir}")
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        shutil.rmtree(persist_dir)
        logging.info(f"Successfully deleted agent {agent_name} and its data at {persist_dir}")
        return {"message": f"Agent '{agent_name}' and its data deleted successfully"}
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logging.error(f"Error deleting agent {agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to an agent and get a response."""
    logging.info(f"Received chat request for agent {request.agent_name}. Message: {request.message}")
    # Normalize agent name consistently with vector store creation
    safe_name = "".join(c if c.isalnum() else "_" for c in request.agent_name).lower()
    persist_dir = os.path.join(AGENTS_DIR, safe_name)
    
    try:
        # We expect the vector store directory to have been created
        # when the agent was initialized from the frontend.
        # If it does not exist, we log a warning and proceed with an empty store.
        collection_name = f"agent_{safe_name}"

        if not os.path.exists(persist_dir):
            logging.warning(
                f"Vector store directory for agent {request.agent_name} "
                f"not found at {persist_dir}. Proceeding with empty store."
            )
            os.makedirs(persist_dir, exist_ok=True)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        retriever = vectorstore.as_retriever()
        
        chat_history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        
        rag_chain = get_rag_chain(retriever, request.agent_name)
        response = rag_chain({"question": request.message, "chat_history": chat_history})
        
        updated_history = request.chat_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response["answer"]}
        ]
        logging.info(f"Agent {request.agent_name} responded to message.")
        return {
            "response": response["answer"],
            "chat_history": updated_history
        }
        
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logging.error(f"Error processing chat for agent {request.agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Include all API routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
