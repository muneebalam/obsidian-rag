import os
import torch
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import hashlib
from datetime import datetime


# Available embedding models configuration
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Fast, good quality embeddings"
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2", 
        "dimensions": 768,
        "description": "High quality embeddings, slower"
    },
    "text-embedding-ada-002": {
        "model_name": "text-embedding-ada-002",
        "dimensions": 1536,
        "description": "OpenAI compatible embeddings"
    },
    "BAAI/bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "description": "BGE small model, good for RAG"
    },
    "BAAI/bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5", 
        "dimensions": 768,
        "description": "BGE base model, high quality"
    },
    "roberta-base": {
        "model_name": "sentence-transformers/roberta-base",
        "dimensions": 768,
        "description": "RoBERTa base model, good general embeddings"
    },
    "roberta-large": {
        "model_name": "sentence-transformers/roberta-large",
        "dimensions": 1024,
        "description": "RoBERTa large model, high quality embeddings"
    },
    "all-roberta-large-v1": {
        "model_name": "sentence-transformers/all-roberta-large-v1",
        "dimensions": 1024,
        "description": "Fine-tuned RoBERTa large for sentence embeddings"
    },
    "paraphrase-multilingual-roberta-base-v2": {
        "model_name": "sentence-transformers/paraphrase-multilingual-roberta-base-v2",
        "dimensions": 768,
        "description": "Multilingual RoBERTa for multiple languages"
    },
    "roberta-base-nli-stsb-mean-tokens": {
        "model_name": "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
        "dimensions": 768,
        "description": "RoBERTa fine-tuned on NLI and STS-B tasks"
    }
}


@st.cache_resource
def load_embedding_model(model_name: str):
    """Load embedding model with caching"""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading embedding model {model_name}: {str(e)}")
        return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def extract_metadata_from_path(file_path: Path, obsidian_dir: Path) -> Dict[str, Any]:
    """Extract metadata from file path"""
    relative_path = file_path.relative_to(obsidian_dir)
    
    return {
        "filename": file_path.name,
        "filepath": str(relative_path),
        "folder": str(relative_path.parent) if relative_path.parent != Path('.') else "root",
        "extension": file_path.suffix,
        "size": file_path.stat().st_size,
        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    }


def process_obsidian_files(obsidian_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Process all markdown files in Obsidian directory"""
    obsidian_dir = Path(obsidian_path)
    
    if not obsidian_dir.exists():
        raise ValueError(f"Obsidian directory not found: {obsidian_path}")
    
    documents = []
    markdown_files = list(obsidian_dir.rglob("*.md"))
    
    if not markdown_files:
        raise ValueError("No markdown files found in the specified directory")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(markdown_files):
        try:
            status_text.text(f"Processing {file_path.name}...")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            metadata = extract_metadata_from_path(file_path, obsidian_dir)
            
            # Chunk the content
            chunks = chunk_text(content, chunk_size, overlap)
            
            # Create documents for each chunk
            for j, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_id": j,
                            "total_chunks": len(chunks)
                        }
                    })
            
            # Update progress
            progress_bar.progress((i + 1) / len(markdown_files))
            
        except Exception as e:
            st.warning(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    status_text.text("Processing complete!")
    progress_bar.empty()
    status_text.empty()
    
    return documents


def embed_documents(documents: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Embed documents using the specified model"""
    # Load embedding model
    model = load_embedding_model(model_name)
    if model is None:
        raise ValueError(f"Failed to load embedding model: {model_name}")
    
    # Extract texts for embedding
    texts = [doc["content"] for doc in documents]
    
    # Generate embeddings
    st.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Create vector database
    vector_db = {
        "model_name": model_name,
        "model_info": EMBEDDING_MODELS[model_name],
        "documents": documents,
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "embedding_dimensions": embeddings.shape[1]
    }
    
    return vector_db


def save_vector_db(vector_db: Dict[str, Any], filepath: str = "vector_db.pkl"):
    """Save vector database to file"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(vector_db, f)
        return True
    except Exception as e:
        st.error(f"Error saving vector database: {str(e)}")
        return False


def load_vector_db(filepath: str = "vector_db.pkl") -> Dict[str, Any]:
    """Load vector database from file"""
    try:
        with open(filepath, 'rb') as f:
            vector_db = pickle.load(f)
        return vector_db
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None


def search_similar_documents(query: str, vector_db: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar documents using cosine similarity"""
    if vector_db is None:
        return []
    
    # Load the same embedding model
    model = load_embedding_model(vector_db["model_name"])
    if model is None:
        return []
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Calculate cosine similarities
    embeddings = vector_db["embeddings"]
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": vector_db["documents"][idx],
            "similarity": float(similarities[idx]),
            "rank": len(results) + 1
        })
    
    return results 