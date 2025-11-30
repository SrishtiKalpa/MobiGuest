from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import routers
from app.routers import chat, scrape, rag

app = FastAPI(
    title="MobiGuest Backend API",
    description="Backend API for MobiGuest Virtual Front Desk Assistant",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(scrape.router, prefix="/api/scrape", tags=["Web Scraping"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])

@app.get("/")
async def root():
    return {"message": "Welcome to MobiGuest Backend API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
