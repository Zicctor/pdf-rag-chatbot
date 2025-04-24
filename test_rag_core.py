#!/usr/bin/env python
# test_rag_core.py - Test the core RAG functionality without API dependencies

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import json

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the functions we want to test
from main import (
    extract_text_from_pdf,
    chunk_text,
    create_embeddings,
    classify_query,
    score_document_relevance,
    factual_retrieval_strategy,
    analytical_retrieval_strategy,
    opinion_retrieval_strategy,
    contextual_retrieval_strategy,
    adaptive_retrieval
)

class MockVectorStore:
    """Mock version of the vector store for testing"""
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
        
    def add_item(self, text, embedding, metadata=None):
        if metadata is None:
            metadata = {}
        metadata["id"] = f"id_{len(self.vectors)}"
        self.vectors.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata)
        return metadata["id"]
        
    def bulk_add_items(self, texts, embeddings, metadatas=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            self.add_item(text, embedding, metadata)
        return len(texts)
        
    def similarity_search(self, query_embedding, k=4, user_id=None, pdf_filename=None):
        results = []
        # Simple cosine similarity
        for i, vector in enumerate(self.vectors):
            # Apply filter if needed
            if user_id and self.metadata[i].get("user_id") != user_id:
                continue
            if pdf_filename and self.metadata[i].get("source") != pdf_filename:
                continue
            
            # Calculate similarity
            similarity = np.dot(query_embedding, vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vector)
            )
            
            results.append({
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "similarity": float(similarity)
            })
            
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
        
    def get_document_count(self, user_id=None, pdf_filename=None):
        count = 0
        for meta in self.metadata:
            if user_id and meta.get("user_id") != user_id:
                continue
            if pdf_filename and meta.get("source") != pdf_filename:
                continue
            count += 1
        return count
        
    def clear_store(self, user_id=None, pdf_filename=None):
        if not user_id and not pdf_filename:
            self.vectors = []
            self.texts = []
            self.metadata = []
            return
            
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

class TestRAGCore(unittest.TestCase):
    """Test class for testing core RAG functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary PDF file for testing
        self.temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.temp_pdf.close()
        
        # Create sample text for chunking
        self.sample_text = "This is a sample text for testing. " * 50
        
        # Create mock vector store
        self.mock_vector_store = MockVectorStore()
        
        # Sample queries for testing
        self.factual_query = "What is the capital of France?"
        self.analytical_query = "Explain the impact of climate change on biodiversity."
        self.opinion_query = "What do people think about remote work?"
        self.contextual_query = "How does this relate to my previous question?"
        
        # Create mock embeddings
        self.mock_embeddings = [
            np.random.rand(768) for _ in range(10)
        ]
        
        # Create mock documents
        self.mock_documents = [
            f"This is document {i} with some information." for i in range(10)
        ]
        
        # Add mock documents to vector store
        for i, (doc, emb) in enumerate(zip(self.mock_documents, self.mock_embeddings)):
            self.mock_vector_store.add_item(
                doc, 
                emb, 
                {"user_id": "test_user", "source": "test.pdf", "index": i}
            )
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_pdf.name):
            os.unlink(self.temp_pdf.name)
    
    def test_chunk_text(self):
        """Test chunking functionality"""
        chunk_size = 100
        overlap = 20
        chunks = chunk_text(self.sample_text, chunk_size, overlap)
        
        # Check that chunks have the expected size and overlap
        self.assertGreater(len(chunks), 1)
        for i in range(len(chunks) - 1):
            # Check chunk size
            if i < len(chunks) - 1:  # Not the last chunk
                self.assertLessEqual(len(chunks[i]), chunk_size)
            
            # Check overlap between consecutive chunks
            if i > 0:
                overlap_text = self.sample_text[(i * (chunk_size - overlap)):(i * (chunk_size - overlap) + overlap)]
                self.assertIn(overlap_text, chunks[i])
    
    @patch('main.embedding_model.encode')
    def test_create_embeddings(self, mock_encode):
        """Test embedding creation"""
        # Mock the encoder to return predictable values
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Test with single text
        embedding = create_embeddings("Test text")
        self.assertEqual(embedding.shape, (3,))
        
        # Test with multiple texts
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        embeddings = create_embeddings(["Text 1", "Text 2"])
        self.assertEqual(embeddings.shape, (2, 3))
    
    @patch('main.client.chat.completions.create')
    def test_classify_query(self, mock_completion):
        """Test query classification"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Factual"
        mock_completion.return_value = mock_response
        
        # Test classification
        query_type = classify_query("What is the population of Tokyo?")
        self.assertEqual(query_type, "Factual")
        
        # Test with different response
        mock_response.choices[0].message.content = "Analytical"
        query_type = classify_query("Analyze the causes of climate change.")
        self.assertEqual(query_type, "Analytical")
    
    @patch('main.client.chat.completions.create')
    def test_factual_retrieval(self, mock_completion):
        """Test factual retrieval strategy"""
        # Setup mock response for query enhancement
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Enhanced factual query"
        mock_completion.return_value = mock_response
        
        # Patch the score_document_relevance function
        with patch('main.score_document_relevance', return_value=0.85):
            # Call the retrieval strategy
            results = factual_retrieval_strategy(
                "Test query", 
                self.mock_vector_store, 
                k=2,
                user_id="test_user", 
                pdf_filename="test.pdf"
            )
            
            # Check results
            self.assertEqual(len(results), 2)
            self.assertIn("text", results[0])
            self.assertIn("metadata", results[0])
            self.assertIn("similarity", results[0])
            self.assertIn("relevance_score", results[0])
    
    @patch('main.client.chat.completions.create')
    def test_analytical_retrieval(self, mock_completion):
        """Test analytical retrieval strategy"""
        # Setup mock response for sub-questions
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Sub-question 1\nSub-question 2\nSub-question 3"
        mock_completion.return_value = mock_response
        
        # Call the retrieval strategy
        results = analytical_retrieval_strategy(
            "Analyze this concept", 
            self.mock_vector_store, 
            k=3,
            user_id="test_user", 
            pdf_filename="test.pdf"
        )
        
        # Check results
        self.assertLessEqual(len(results), 3)
        for result in results:
            self.assertIn("text", result)
            self.assertIn("metadata", result)
            self.assertIn("similarity", result)
    
    @patch('main.client.chat.completions.create')
    def test_adaptive_retrieval(self, mock_completion):
        """Test adaptive retrieval which uses classification to choose strategy"""
        # Setup mock response for classification
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Factual"
        mock_completion.return_value = mock_response
        
        # Mock the specific retrieval strategy
        with patch('main.factual_retrieval_strategy') as mock_strategy:
            mock_strategy.return_value = [{"text": "result", "metadata": {}, "similarity": 0.9}]
            
            # Call adaptive retrieval
            results = adaptive_retrieval(
                "Test query", 
                self.mock_vector_store, 
                k=1,
                user_id="test_user", 
                pdf_filename="test.pdf"
            )
            
            # Check results
            self.assertEqual(len(results), 1)
            mock_strategy.assert_called_once()

if __name__ == '__main__':
    unittest.main() 