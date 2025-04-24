# PDF RAG Chatbot with Adaptive Retrieval

A PDF Question Answering system that uses an adaptive retrieval approach based on query type classification. The system uses an in-memory vector store for simplicity and ease of deployment.

## Features

- Upload PDF documents for processing
- Automatic classification of queries into Factual, Analytical, Opinion, or Contextual types
- Specialized retrieval strategies for each query type
- API for question answering against uploaded PDFs
- Document management system with user-specific collections
- Simple in-memory vector storage for quick setup

## Architecture

- FastAPI backend for REST API endpoints
- Sentence Transformer (`all-mpnet-base-v2`) for generating embeddings
- Llama 3.3 70B Instruct Turbo for LLM tasks (via Together.ai)
- Custom SimpleVectorStore for in-memory vector storage
- PyMuPDF (fitz) for PDF text extraction

## Setup

### Prerequisites

- Python 3.9+ 
- Together.ai API key (for LLM access)

### Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Together.ai API key:
   ```
   OPENAI_API_KEY=your-together-ai-key
   ```

## Running the Application

Start the FastAPI server:

```
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

The web interface will be available at `http://localhost:8000`. You can upload PDFs and ask questions about them through this interface.

## API Endpoints

### Upload a PDF

```
POST /upload_pdf/
```

- Request: Form data with file
- Response: JSON with filename and user_id

### Query a PDF

```
POST /rag_query/
```

- Request Body:
  ```json
  {
    "pdf_filename": "example.pdf",
    "query": "What is AI?",
    "k": 4,
    "user_context": "I'm a beginner",
    "chunk_size": 1000,
    "user_id": "optional-user-id"
  }
  ```
- Response: JSON with query results, including retrieved documents and generated response

### List User Documents

```
GET /user_documents/
```

- Response: JSON with list of user's uploaded documents

### Delete Document

```
DELETE /documents/{filename}
```

- Response: JSON with deletion confirmation

## Retrieval Strategies

The system adaptively selects a retrieval strategy based on query classification:

1. **Factual Strategy**: For precise factual information, uses query enhancement and relevance scoring.
2. **Analytical Strategy**: For comprehensive analysis, breaks down complex queries into sub-questions.
3. **Opinion Strategy**: For opinion-based queries, identifies and retrieves diverse perspectives.
4. **Contextual Strategy**: For context-dependent queries, reformulates the question based on user context.

## Important Notes

- This implementation uses an in-memory vector store, so all data is lost when the server restarts
- For production use, consider implementing a persistent vector store solution
- The `SimpleVectorStore` class in `main.py` can be extended to support persistence if needed

## Customization

- Modify the `.env` file to adjust API settings
- Change the embedding model or LLM in `main.py`
- Adjust retrieval parameters like chunk size in API requests
