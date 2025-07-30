import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional
import gc
import os
from pathlib import Path
from src.app import MODELS_CONFIG, load_model_and_tokenizer
from src.app.sidebar import sidebar_page
from src.utils import generate_response



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

sidebar_page()