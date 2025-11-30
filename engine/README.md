# MobiGuest Backend

This is the backend service for the MobiGuest Virtual Front Desk Assistant, built with FastAPI.

## Features

- RESTful API endpoints for chat functionality
- Web scraping capabilities
- RAG (Retrieval-Augmented Generation) pipeline
- Vector store for document storage and retrieval

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MobiGuest/backend
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the backend directory with the following variables:
   ```
   OLLAMA_API_BASE=http://localhost:11434
   ```

## Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Chat
- `POST /api/chat/chat` - Send a chat message and get a response

### Web Scraping
- `POST /api/scrape/scrape` - Scrape a website and return its content

### RAG
- `POST /api/rag/query` - Query the RAG system
- `POST /api/rag/load_documents` - Load documents into the vector store

## Development

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

### Code Style
This project uses `black` for code formatting and `isort` for import sorting.

```bash
# Install pre-commit hooks
pre-commit install

# Run formatters
black .
isort .
```

## License
[MIT](LICENSE)
