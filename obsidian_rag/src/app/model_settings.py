import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils.config import load_config, get_api_key
from src.utils.models import MODELS_CONFIG


@st.cache_resource
def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, None


def get_available_models(config=None):
    """Get list of available models based on API key availability"""
    if config is None:
        config = load_config()
    
    available_models = []
    
    for model_name, model_config in MODELS_CONFIG.items():
        if model_config["type"] == "local":
            # Local models are always available
            available_models.append(model_name)
        elif model_config["type"] == "api":
            # Check if API key is available
            provider = model_config["provider"]
            api_key = get_api_key(provider, config)
            if api_key and api_key.strip():
                available_models.append(model_name)
    
    return available_models


def validate_model_access(model_name: str, config=None):
    """Validate if a model can be accessed (API key available for API models)"""
    if model_name not in MODELS_CONFIG:
        return False, "Model not found"
    
    model_config = MODELS_CONFIG[model_name]
    
    if model_config["type"] == "local":
        return True, "Local model available"
    elif model_config["type"] == "api":
        provider = model_config["provider"]
        api_key = get_api_key(provider, config)
        if api_key and api_key.strip():
            return True, f"API key available for {provider}"
        else:
            return False, f"No API key found for {provider}"
    
    return False, "Unknown model type"