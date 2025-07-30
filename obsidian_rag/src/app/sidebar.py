import streamlit as st
from src.app import MODELS_CONFIG, load_model_and_tokenizer
import torch
import gc

def sidebar_page():
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