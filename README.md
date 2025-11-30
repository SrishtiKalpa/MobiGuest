# MobiGuest - AI-Powered Hotel Assistant

MobiGuest is an intelligent assistant designed to handle hotel guest inquiries, bookings, and support through a user-friendly web interface.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Ollama (for local embeddings)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MobiGuest
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Download and install from [Ollama's official website](https://ollama.ai/)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - In a new terminal, pull the required model:
     ```bash
     ollama pull nomic-embed-text
     ```

## Running the Application

The application consists of two main components:

### 1. Backend Server

```bash
cd backend
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/api/docs`
- Interactive API Docs: `http://localhost:8000/api/redoc`

### 2. Frontend (Streamlit)

In a new terminal:

```bash
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

## Using the Application

1. **Access the Web Interface**
   - Open your browser and go to `http://localhost:8501`

2. **Available Agents**
   - The application comes with a default agent called "HotelBot"
   - You can create new agents as needed

3. **Chat with an Agent**
   - Select an agent from the list
   - Type your message in the chat input
   - The AI will respond based on its training and knowledge

## Troubleshooting

### Port Already in Use
If you get a port conflict:

```bash
# Find the process using the port (e.g., 8000)
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Missing Dependencies
If you encounter missing package errors:

```bash
pip install -r requirements.txt
```

### Ollama Connection Issues
Ensure Ollama is running:

```bash
ollama serve
```

In a separate terminal, verify the model is available:
```bash
ollama list
```

## Project Structure

```
MobiGuest/
├── backend/               # FastAPI backend
│   ├── main.py           # Main application file
│   ├── requirements.txt  # Backend dependencies
│   └── chroma_db/        # Vector database storage
├── front_desk_bot/       # Frontend components
│   └── requirements.txt  # Frontend dependencies
├── app.py               # Streamlit frontend
└── README.md            # This file
```

## API Endpoints

- `POST /api/chat` - Send a message to an agent
- `GET /api/agents` - List all available agents
- `POST /api/agents` - Create a new agent

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.
