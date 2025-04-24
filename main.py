import os 
import numpy as np
import json
import fitz
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import uuid

# Simple in-memory vector store implementation
class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store original texts
        self.metadata = []  # List to store metadata for each text
    
    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
        text (str): The original text.
        embedding (List[float]): The embedding vector.
        metadata (dict, optional): Additional metadata.
        """
        if metadata is None:
            metadata = {}
            
        # Ensure user_id and source are in metadata
        if "user_id" not in metadata:
            metadata["user_id"] = str(uuid.uuid4())
        if "source" not in metadata:
            metadata["source"] = "unknown"
            
        # Generate a unique ID for this vector if not present
        if "id" not in metadata:
            metadata["id"] = str(uuid.uuid4())
            
        self.vectors.append(np.array(embedding))  # Convert embedding to numpy array and add to vectors list
        self.texts.append(text)  # Add the original text to texts list
        self.metadata.append(metadata)  # Add metadata to metadata list
        
        return metadata["id"]
    
    def bulk_add_items(self, texts, embeddings, metadatas=None):
        """
        Add multiple items to the vector store in batch.
        
        Args:
            texts (List[str]): List of text contents
            embeddings (List[np.ndarray]): List of embeddings
            metadatas (List[dict], optional): List of metadata dicts
            
        Returns:
            int: Number of items added
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Add index to metadata
            metadata["index"] = i
            self.add_item(text, embedding, metadata)
            
        return len(texts)
    
    def similarity_search(self, query_embedding, k=4, user_id=None, pdf_filename=None):
        """
        Find similar items to the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to return
            user_id (str, optional): Filter by user ID
            pdf_filename (str, optional): Filter by PDF filename
            
        Returns:
            List[Dict]: List of similar items with metadata
        """
        # Create a filter function based on user_id and pdf_filename
        def filter_func(metadata):
            if user_id and metadata.get("user_id") != user_id:
                return False
            if pdf_filename and metadata.get("source") != pdf_filename:
                return False
            return True
            
        if not self.vectors:
            return []  # Return empty list if no vectors are stored
        
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Apply filter
            if not filter_func(self.metadata[i]):
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the text
                "metadata": self.metadata[idx],  # Add the metadata
                "similarity": float(score)  # Add the similarity score
            })
        
        return results
    
    def get_document_count(self, user_id=None, pdf_filename=None):
        """
        Get count of documents for a user and/or PDF.
        
        Args:
            user_id (str, optional): User ID to filter by
            pdf_filename (str, optional): PDF filename to filter by
            
        Returns:
            int: Count of matching documents
        """
        count = 0
        for meta in self.metadata:
            if user_id and meta.get("user_id") != user_id:
                continue
            if pdf_filename and meta.get("source") != pdf_filename:
                continue
            count += 1
        return count
    
    def clear_store(self, user_id=None, pdf_filename=None):
        """
        Delete documents matching the filters.
        
        Args:
            user_id (str, optional): User ID to filter by
            pdf_filename (str, optional): PDF filename to filter by
        """
        if not user_id and not pdf_filename:
            # Clear all items
            self.vectors = []
            self.texts = []
            self.metadata = []
            return
        
        # Create new lists without the filtered items
        new_vectors = []
        new_texts = []
        new_metadata = []
        
        for i, meta in enumerate(self.metadata):
            should_keep = True
            
            if user_id and meta.get("user_id") == user_id:
                should_keep = False
            if pdf_filename and meta.get("source") == pdf_filename:
                should_keep = False
                
            if should_keep:
                new_vectors.append(self.vectors[i])
                new_texts.append(self.texts[i])
                new_metadata.append(meta)
                
        self.vectors = new_vectors
        self.texts = new_texts
        self.metadata = new_metadata

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI/Together API client for LLM completions (not embeddings)
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load HuggingFace embedding model once
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Create a global instance of the vector store
_vector_store = None

# Function to get the vector store
def get_vector_store():
    """Returns a vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = SimpleVectorStore()
    return _vector_store

# Extracting text from a PDF file

def extract_text_from_pdf(pdf_path):
    #Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = "" # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num] # Get the page
        text = page.get_text("text") # Extract text from the page
        all_text += text # Append the extracted text to the all_text string

    return all_text

# Chunking the extracted text
def chunk_text(text, n, overlap):
    chunks = [] # Intialize an empty list to store the chunks

    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])
    
    return chunks # Return the list of text chunks

def create_embeddings(text):
    # Handle both string and list inputs by converting string input a list
    input_text = text if isinstance(text, list) else [text]
    embeddings = embedding_model.encode(input_text, show_progress_bar=False)
    if isinstance(text, str):
        return embeddings[0]
    return embeddings

def process_document(pdf_path, chunk_size, user_id=None, chunk_overlap=200):
    # Use a unique user ID if not provided
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    # Extract the filename from the path
    pdf_filename = os.path.basename(pdf_path)

    # Extract text from the PDF file
    print("Extracting text from the pdf ...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Chunk the extracted text
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")

    # Create embeddings for the text chunks
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)

    # Initialize the Pinecone vector store
    store = get_vector_store()

    # Prepare metadata for bulk insertion
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({
            "user_id": user_id,
            "source": pdf_filename,
            "index": i
        })

    # Add the text chunks and their embeddings to the vector store in bulk
    print("Adding chunks to vector store...")
    store.bulk_add_items(chunks, chunk_embeddings, metadatas)
    
    print(f"Added {len(chunks)} chunks to the vector store for user {user_id}")

    # Return the user_id (for tracking) and the vector store
    return user_id, store

# Implement specialized Retrieval strategies with Pinecone support

# Factual Strategy
def factual_retrieval_strategy(query, vector_store, k=4, user_id=None, pdf_filename=None):
    """
    Retrieval strategy for factual queries focusing on precision.
    
    Args:
        query (str): User query
        vector_store (PineconeVectorStore): Vector store
        k (int): Number of documents to return
        user_id (str, optional): User ID to filter results
        pdf_filename (str, optional): PDF filename to filter results
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Factual retrieval strategy for: '{query}'")
    
    # Use LLM to enhance the query for better precision
    system_prompt = """You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and 
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """

    user_prompt = f"Enhance this factual query: {query}"
    
    # Generate the enhanced query using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and print the enhanced query
    enhanced_query = response.choices[0].message.content.strip()
    print(f"Enhanced query: {enhanced_query}")
    
    # Create embeddings for the enhanced query
    query_embedding = create_embeddings(enhanced_query)
    
    # Perform initial similarity search to retrieve documents
    initial_results = vector_store.similarity_search(
        query_embedding, 
        k=k*2, 
        user_id=user_id, 
        pdf_filename=pdf_filename
    )
    
    # Initialize a list to store ranked results
    ranked_results = []
    
    # Score and rank documents by relevance using LLM
    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    
    # Sort the results by relevance score in descending order
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return the top k results
    return ranked_results[:k]

# Analytical Strategy
def analytical_retrieval_strategy(query, vector_store, k=4, user_id=None, pdf_filename=None):
    """
    Retrieval strategy for analytical queries focusing on comprehensive coverage.
    
    Args:
        query (str): User query
        vector_store (PineconeVectorStore): Vector store
        k (int): Number of documents to return
        user_id (str, optional): User ID to filter results
        pdf_filename (str, optional): PDF filename to filter results
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Analytical retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in generating sub-questions
    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """

    # Create the user prompt with the main query
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    # Generate the sub-questions using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # Extract and clean the sub-questions
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"Generated sub-queries: {sub_queries}")
    
    # Retrieve documents for each sub-query
    all_results = []
    for sub_query in sub_queries:
        # Create embeddings for the sub-query
        sub_query_embedding = create_embeddings(sub_query)
        # Perform similarity search for the sub-query
        results = vector_store.similarity_search(
            sub_query_embedding, 
            k=2, 
            user_id=user_id, 
            pdf_filename=pdf_filename
        )
        all_results.extend(results)
    
    # Ensure diversity by selecting from different sub-query results
    # Remove duplicates (same document ID)
    unique_docs = {}
    for result in all_results:
        doc_id = result["metadata"].get("id")
        if doc_id not in unique_docs:
            unique_docs[doc_id] = result
    
    diverse_results = list(unique_docs.values())
    
    # If we need more results to reach k, add more from initial results
    if len(diverse_results) < k:
        # Direct retrieval for the main query
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(
            main_query_embedding, 
            k=k, 
            user_id=user_id, 
            pdf_filename=pdf_filename
        )
        
        for result in main_results:
            doc_id = result["metadata"].get("id")
            if doc_id not in unique_docs and len(diverse_results) < k:
                unique_docs[doc_id] = result
                diverse_results.append(result)
    
    # Return the top k diverse results
    return diverse_results[:k]

# Opinion Strategy
def opinion_retrieval_strategy(query, vector_store, k=4, user_id=None, pdf_filename=None):
    """
    Retrieval strategy for opinion queries focusing on diverse perspectives.
    
    Args:
        query (str): User query
        vector_store (PineconeVectorStore): Vector store
        k (int): Number of documents to return
        user_id (str, optional): User ID to filter results
        pdf_filename (str, optional): PDF filename to filter results
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Opinion retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in identifying different perspectives
    system_prompt = """You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives 
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
    """

    # Create the user prompt with the main query
    user_prompt = f"Identify different perspectives on: {query}"
    
    # Generate the different perspectives using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # Extract and clean the viewpoints
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"Identified viewpoints: {viewpoints}")
    
    # Retrieve documents representing each viewpoint
    all_results = []
    for viewpoint in viewpoints:
        # Combine the main query with the viewpoint
        combined_query = f"{query} {viewpoint}"
        # Create embeddings for the combined query
        viewpoint_embedding = create_embeddings(combined_query)
        # Perform similarity search for the combined query
        results = vector_store.similarity_search(
            viewpoint_embedding, 
            k=2, 
            user_id=user_id, 
            pdf_filename=pdf_filename
        )
        
        # Mark results with the viewpoint they represent
        for result in results:
            result["viewpoint"] = viewpoint
        
        # Add the results to the list of all results
        all_results.extend(results)
    
    # Select a diverse range of opinions
    # Ensure we get at least one document from each viewpoint if possible
    selected_results = []
    for viewpoint in viewpoints:
        # Filter documents by viewpoint
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    
    # Fill remaining slots with highest similarity docs
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # Sort remaining docs by similarity
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    
    # Return the top k results
    return selected_results[:k]

# Contextual Strategy
def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None, user_id=None, pdf_filename=None):
    """
    Retrieval strategy for contextual queries integrating user context.
    
    Args:
        query (str): User query
        vector_store (PineconeVectorStore): Vector store
        k (int): Number of documents to return
        user_context (str): Additional user context
        user_id (str, optional): User ID to filter results
        pdf_filename (str, optional): PDF filename to filter results
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Contextual retrieval strategy for: '{query}'")
    
    # If no user context provided, try to infer it from the query
    if not user_context:
        system_prompt = """You are an expert at understanding implied context in questions.
For the given query, infer what contextual information might be relevant or implied 
but not explicitly stated. Focus on what background would help answering this query.

Return a brief description of the implied context."""

        user_prompt = f"Infer the implied context in this query: {query}"
        
        # Generate the inferred context using the LLM
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # Extract and print the inferred context
        user_context = response.choices[0].message.content.strip()
        print(f"Inferred context: {user_context}")
    
    # Reformulate the query to incorporate context
    system_prompt = """You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that 
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation."""

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:"""
    
    # Generate the contextualized query using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and print the contextualized query
    contextualized_query = response.choices[0].message.content.strip()
    print(f"Contextualized query: {contextualized_query}")
    
    # Retrieve documents based on the contextualized query
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(
        query_embedding, 
        k=k*2, 
        user_id=user_id, 
        pdf_filename=pdf_filename
    )
    
    # Rank documents considering both relevance and user context
    ranked_results = []
    
    for doc in initial_results:
        # Score document relevance considering the context
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    
    # Sort by context relevance and return top k results
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]

# The core adaptive retriever
def adaptive_retrieval(query, vector_store, k=4, user_context=None, user_id=None, pdf_filename=None):
    """
    Perform adaptive retrieval by selecting and executing the appropriate strategy.
    
    Args:
        query (str): User query
        vector_store (PineconeVectorStore): Vector store
        k (int): Number of documents to retrieve
        user_context (str): Optional user context for contextual queries
        user_id (str): Optional user ID to filter results
        pdf_filename (str): Optional PDF filename to filter results
        
    Returns:
        List[Dict]: Retrieved documents
    """
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Select and execute the appropriate retrieval strategy based on the query type
    if query_type == "Factual":
        # Use the factual retrieval strategy for precise information
        results = factual_retrieval_strategy(query, vector_store, k, user_id, pdf_filename)
    elif query_type == "Analytical":
        # Use the analytical retrieval strategy for comprehensive coverage
        results = analytical_retrieval_strategy(query, vector_store, k, user_id, pdf_filename)
    elif query_type == "Opinion":
        # Use the opinion retrieval strategy for diverse perspectives
        results = opinion_retrieval_strategy(query, vector_store, k, user_id, pdf_filename)
    elif query_type == "Contextual":
        # Use the contextual retrieval strategy, incorporating user context
        results = contextual_retrieval_strategy(query, vector_store, k, user_context, user_id, pdf_filename)
    else:
        # Default to factual retrieval strategy if classification fails
        results = factual_retrieval_strategy(query, vector_store, k, user_id, pdf_filename)
    
    return results  # Return the retrieved documents

def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None, chunk_size=1000, user_id=None):
    """
    Complete RAG pipeline with adaptive retrieval and vector store.
    
    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        k (int): Number of documents to retrieve
        user_context (str): Optional user context
        chunk_size (int): Size of chunks for text processing
        user_id (str): Optional user ID for personalization
    
    Returns:
        Dict: Results including query, retrieved documents, query type, and response
    """
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    print(f"Query: {query}")
    
    # Get the PDF filename
    pdf_filename = os.path.basename(pdf_path)
    
    # Initialize the vector store
    vector_store = get_vector_store()
    
    # Check if this PDF has already been processed for this user
    document_count = vector_store.get_document_count(user_id=user_id, pdf_filename=pdf_filename)
    
    # If this PDF hasn't been processed yet for this user, process it
    if document_count == 0:
        # Process the document to extract text, chunk it, create embeddings, and store
        user_id, _ = process_document(pdf_path, chunk_size, user_id)
    else:
        print(f"Using existing vectors for PDF '{pdf_filename}' (User: {user_id})")
    
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Retrieve documents using the adaptive retrieval strategy based on the query type
    retrieved_docs = adaptive_retrieval(
        query, 
        vector_store, 
        k, 
        user_context, 
        user_id=user_id, 
        pdf_filename=pdf_filename
    )
    
    # Generate a response based on the query, retrieved documents, and query type
    response = generate_response(query, retrieved_docs, query_type)
    
    # Compile the results into a dictionary
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response,
        "user_id": user_id
    }
    
    print("\n=== RESPONSE ===")
    print(response)
    
    return result

# Helper Functions for Document Scoring
def score_document_relevance(query, document, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """
    Score document relevance to a query using LLM.
    
    Args:
        query (str): User query
        document (str): Document text
        model (str): LLM model
        
    Returns:
        float: Relevance score from 0-10
    """
    # System prompt to instruct the model on how to rate relevance
    system_prompt = """You are an expert at evaluating document relevance.
        Rate the relevance of a document to a query on a scale from 0 to 10, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # Truncate document if it's too long
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # User prompt containing the query and document preview
    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """
    
    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract the score from the model's response
    score_text = response.choices[0].message.content.strip()
    
    # Extract numeric score using regex
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        # Default score if extraction fails
        return 5.0

def score_document_context_relevance(query, context, document, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """
    Score document relevance considering both query and context.
    
    Args:
        query (str): User query
        context (str): User context
        document (str): Document text
        model (str): LLM model
        
    Returns:
        float: Relevance score from 0-10
    """
    # System prompt to instruct the model on how to rate relevance considering context
    system_prompt = """You are an expert at evaluating document relevance considering context.
        Rate the document on a scale from 0 to 10 based on how well it addresses the query
        when considering the provided context, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query in the given context

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # Truncate document if it's too long
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # User prompt containing the query, context, and document preview
    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    Relevance score considering context (0-10):
    """
    
    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract the score from the model's response
    score_text = response.choices[0].message.content.strip()
    
    # Extract numeric score using regex
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        # Default score if extraction fails
        return 5.0

def classify_query(query, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """
    Classify a query into one of four categories: Factual, Analytical, Opinion, or Contextual.
    
    Args:
        query (str): User query
        model (str): LLM model to use
        
    Returns:
        str: Query category
    """
    # Define the system prompt to guide the AI's classification
    system_prompt = """You are an expert at classifying questions. 
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """

    # Create the user prompt with the query to be classified
    user_prompt = f"Classify this query: {query}"
    
    # Generate the classification response from the AI model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and strip the category from the response
    category = response.choices[0].message.content.strip()
    
    # Define the list of valid categories
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    
    # Ensure the returned category is valid
    for valid in valid_categories:
        if valid in category:
            return valid
    
    # Default to "Factual" if classification fails
    return "Factual"

def generate_response(query, results, query_type, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """
    Generate a response based on query, retrieved documents, and query type.
    
    Args:
        query (str): User query
        results (List[Dict]): Retrieved documents
        query_type (str): Type of query
        model (str): LLM model
        
    Returns:
        str: Generated response
    """
    # Prepare context from retrieved documents by joining their texts with separators
    context = "\n\n---\n\n".join([r["text"] for r in results])
    
    # Create custom system prompt based on query type
    if query_type == "Factual":
        system_prompt = """You are a helpful assistant providing factual information.
    Answer the question based on the provided context. Focus on accuracy and precision.
    If the context doesn't contain the information needed, acknowledge the limitations."""
        
    elif query_type == "Analytical":
        system_prompt = """You are a helpful assistant providing analytical insights.
    Based on the provided context, offer a comprehensive analysis of the topic.
    Cover different aspects and perspectives in your explanation.
    If the context has gaps, acknowledge them while providing the best analysis possible."""
        
    elif query_type == "Opinion":
        system_prompt = """You are a helpful assistant discussing topics with multiple viewpoints.
    Based on the provided context, present different perspectives on the topic.
    Ensure fair representation of diverse opinions without showing bias.
    Acknowledge where the context presents limited viewpoints."""
        
    elif query_type == "Contextual":
        system_prompt = """You are a helpful assistant providing contextually relevant information.
    Answer the question considering both the query and its context.
    Make connections between the query context and the information in the provided documents.
    If the context doesn't fully address the specific situation, acknowledge the limitations."""
        
    else:
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."""
    
    # Create user prompt by combining the context and the query
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """
    
    # Generate response using the OpenAI client
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    # Return the generated response content
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== RUNNING ADAPTIVE RAG PIPELINE ===")
    pdf_path = "data/AI_Information.pdf"  # Update as needed
    test_query = "What is Explainable AI (XAI)?"  # Example query, replace as desired
    results = rag_with_adaptive_retrieval(pdf_path, test_query, k=4, chunk_size=1000)
    print("\n=== ADAPTIVE RAG RESULTS ===")
    print(f"\nQuery: {results['query']}")
    print(f"Query Type: {results['query_type']}")
    print("\nTop Retrieved Documents:")
    for i, doc in enumerate(results['retrieved_documents']):
        print(f"[{i+1}] {doc['text'][:200]}...\n")
    print("\nGenerated Response:")
    print(results['response'])