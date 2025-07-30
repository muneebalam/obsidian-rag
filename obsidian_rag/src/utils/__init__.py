# Import specific functions to avoid circular imports
from .chat import generate_response
from .embeddings import (
    EMBEDDING_MODELS, 
    load_embedding_model, 
    process_obsidian_files, 
    embed_documents, 
    save_vector_db, 
    load_vector_db,
    search_similar_documents
)
from .config import (
    load_config, 
    get_api_key, 
    get_model_config, 
    validate_api_keys
)
from .api_chat import (
    generate_gemini_response,
    generate_api_response
)
from .models import *