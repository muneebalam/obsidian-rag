import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional
import gc
import os
from pathlib import Path
from src.app import MODELS_CONFIG, load_model_and_tokenizer

def generate_response(prompt: str, tokenizer, model, model_config: dict) -> str:
    """Generate response using the selected model"""
    try:
        # Prepare input based on model type
        if "dialo" in model_config["model_name"].lower():
            # For DialoGPT models
            inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        elif "tinyllama" in model_config["model_name"].lower():
            # For TinyLlama models
            inputs = tokenizer.encode(f"<|system|>You are a helpful assistant.</s><|user|>{prompt}</s><|assistant|>", return_tensors='pt')
        else:
            # For other models (GPT-2, BLOOM, etc.)
            inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + model_config["max_length"],
                temperature=model_config["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode response
        if "dialo" in model_config["model_name"].lower():
            response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        elif "tinyllama" in model_config["model_name"].lower():
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].split("</s>")[0].strip()
        else:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
        
        return response if response else "I'm not sure how to respond to that."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Sidebar for model selection
st.sidebar.title("🤖 Settings")



# Obsidian directory path input
st.sidebar.subheader("📁 Obsidian Directory")
obsidian_path = st.sidebar.text_input(
    "Path to Obsidian vault:",
    value="/Users/muneebalam/Documents/DS docs/",
    help="Enter the full path to your Obsidian vault directory"
)

st.sidebar.subheader("🤖 Chat Model")

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "Choose a Model:",
    list(MODELS_CONFIG.keys()),
    index=0
)

# Get selected model config
selected_config = MODELS_CONFIG[selected_model_name]

# Model parameters
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, selected_config["temperature"], 0.1)
max_length = st.sidebar.slider("Max Length", 50, 200, selected_config["max_length"], 10)

# Update config with sidebar values
selected_config["temperature"] = temperature
selected_config["max_length"] = max_length

# Load model if changed
if (st.session_state.current_model != selected_config["model_name"] or 
    st.session_state.tokenizer is None or 
    st.session_state.model is None):
    
    with st.spinner(f"Loading {selected_model_name}..."):
        # Clear previous model from memory
        if st.session_state.model is not None:
            del st.session_state.model
            del st.session_state.tokenizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load new model
        tokenizer, model = load_model_and_tokenizer(selected_config["model_name"])
        
        if tokenizer is not None and model is not None:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.current_model = selected_config["model_name"]
            st.sidebar.success(f"✅ {selected_model_name} loaded successfully!")
        else:
            st.sidebar.error(f"❌ Failed to load {selected_model_name}")

# Display model info
if st.session_state.current_model:
    st.sidebar.subheader("Current Model")
    st.sidebar.info(f"**Model:** {selected_model_name}\n\n**Path:** {st.session_state.current_model}")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Page selection
st.sidebar.markdown("---")