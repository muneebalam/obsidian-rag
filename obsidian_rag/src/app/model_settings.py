# Available models configuration
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODELS_CONFIG = {
    "GPT-2 (Small)": {
        "model_name": "gpt2",
        "max_length": 100,
        "temperature": 0.7
    },
    "GPT-2 (Medium)": {
        "model_name": "gpt2-medium",
        "max_length": 100,
        "temperature": 0.7
    },
    "BLOOM (560M)": {
        "model_name": "bigscience/bloom-560m",
        "max_length": 100,
        "temperature": 0.7
    },
    "DialoGPT (Small)": {
        "model_name": "microsoft/DialoGPT-small",
        "max_length": 100,
        "temperature": 0.7
    },
    "DialoGPT (Medium)": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 100,
        "temperature": 0.7
    },
    "TinyLlama": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_length": 150,
        "temperature": 0.7
    }
}

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