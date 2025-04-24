from fastapi import FastAPI, UploadFile, File, Form, Header, Cookie, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import numpy as np
from typing import Optional, Dict, List, Any
import logging

# Import the main logic from your existing script
from main import rag_with_adaptive_retrieval, get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF RAG Chatbot API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    pdf_filename: str
    query: str
    k: Optional[int] = 4
    user_context: Optional[str] = None
    chunk_size: Optional[int] = 1000
    user_id: Optional[str] = None

def convert_numpy_to_python(obj: Any) -> Any:
    """Convert NumPy types to standard Python types to make them JSON serializable."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return convert_numpy_to_python(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(np, 'complex') and isinstance(obj, np.complex):
        return complex(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        try:
            # Try to convert to dict/list/simple type
            return convert_numpy_to_python(dict(obj))
        except (TypeError, ValueError):
            try:
                return str(obj)
            except:
                return "Unserializable object"

def get_user_id(x_user_id: Optional[str] = Header(None), user_id_cookie: Optional[str] = Cookie(None)):
    """Get or generate a user ID from headers, cookies, or create a new one."""
    user_id = x_user_id or user_id_cookie
    if not user_id:
        user_id = str(uuid.uuid4())
        logger.info(f"Generated new user ID: {user_id}")
    return user_id

@app.post("/upload_pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id)
):
    try:
        logger.info(f"Received file upload: {file.filename}, content_type: {file.content_type} for user: {user_id}")
        
        # Verify it's a PDF file
        if not file.content_type.startswith("application/pdf"):
            return JSONResponse(
                status_code=400, 
                content={"detail": f"File must be a PDF, got {file.content_type}"}
            )
            
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Saving file to: {file_location}")
        
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Verify the file was saved correctly
        if os.path.exists(file_location):
            file_size = os.path.getsize(file_location)
            logger.info(f"File saved successfully. Size: {file_size} bytes")
            
            # Check if the file has already been processed by this user
            vector_store = get_vector_store()
            document_count = vector_store.get_document_count(user_id=user_id, pdf_filename=file.filename)
            
            if document_count > 0:
                logger.info(f"Using existing vectors for PDF '{file.filename}' (User: {user_id})")
            else:
                # Process the file to extract text, create chunks and embeddings
                logger.info(f"Processing PDF for indexing: {file.filename}")
                # Note: We're not actually processing here - we'll do it lazily on the first query
                
            return {
                "filename": file.filename, 
                "message": "PDF uploaded successfully.",
                "user_id": user_id
            }
        else:
            return JSONResponse(
                status_code=500, 
                content={"detail": "Failed to save the file"}
            )
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"detail": f"Error uploading file: {str(e)}"}
        )

@app.post("/rag_query/")
def rag_query(
    request: QueryRequest,
    user_id: str = Depends(get_user_id)
):
    try:
        # Override user_id in request if provided in header/cookie
        if not request.user_id:
            request.user_id = user_id
            
        logger.info(f"Received query for PDF '{request.pdf_filename}' from user {request.user_id}: {request.query}")
        
        pdf_path = os.path.join(UPLOAD_DIR, request.pdf_filename)
        if not os.path.exists(pdf_path):
            return JSONResponse(
                status_code=404, 
                content={"error": "PDF file not found. Please upload first."}
            )
        
        # Run the RAG query with user tracking
        result = rag_with_adaptive_retrieval(
            pdf_path=pdf_path,
            query=request.query,
            k=request.k,
            user_context=request.user_context,
            chunk_size=request.chunk_size,
            user_id=request.user_id
        )
        
        # Convert NumPy values to standard Python types
        serializable_result = convert_numpy_to_python(result)
        
        return serializable_result
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing query: {str(e)}"}
        )

@app.get("/user_documents/")
def get_user_documents(user_id: str = Depends(get_user_id)):
    """Get a list of documents uploaded by the user."""
    try:
        vector_store = get_vector_store()
        
        # Query documents from Pinecone
        try:
            # For Pinecone, we use a different approach since it doesn't support SQL-like queries
            # We'll use document_count to check if documents exist for this user
            # This is a simplified approach - in a real app, you might want to enhance this
            
            # Look through the files in the upload directory
            documents = []
            for filename in os.listdir(UPLOAD_DIR):
                if filename.endswith(".pdf"):
                    # Check if this PDF has vectors in Pinecone for this user
                    count = vector_store.get_document_count(user_id=user_id, pdf_filename=filename)
                    if count > 0:
                        documents.append({
                            "filename": filename,
                            "chunk_count": count
                        })
            
            return {"user_id": user_id, "documents": documents}
        except Exception as e:
            logger.error(f"Error querying Pinecone for documents: {str(e)}")
            return {"user_id": user_id, "documents": []}
    except Exception as e:
        logger.error(f"Error getting user documents: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting user documents: {str(e)}"}
        )

@app.delete("/documents/{filename}")
def delete_document(filename: str, user_id: str = Depends(get_user_id)):
    """Delete a document and its embeddings for a specific user."""
    try:
        vector_store = get_vector_store()
        vector_store.clear_store(user_id=user_id, pdf_filename=filename)
        
        # Also delete the file if it exists
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        
        return {"message": f"Document '{filename}' deleted for user {user_id}"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting document: {str(e)}"}
        )

@app.get("/")
def root():
    return {"message": "Welcome to the PDF RAG Chatbot FastAPI!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
