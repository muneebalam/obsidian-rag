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


def initialize_session_state():
    """Initialize session state variables"""
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


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Get sidebar configuration
    page, selected_model_name, selected_config = sidebar_page()

    # Main content area
    if page == "💬 Chat":
        # Chat interface
        st.title("🤖 AI Chatbot")
        st.markdown(f"**Current Model:** {selected_model_name}")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                if st.session_state.tokenizer is not None and st.session_state.model is not None:
                    with st.spinner("Generating response..."):
                        response = generate_response(
                            prompt, 
                            st.session_state.tokenizer, 
                            st.session_state.model, 
                            selected_config
                        )
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                else:
                    error_msg = "Model not loaded. Please select a model from the sidebar."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)

        # Footer
        st.markdown("---")
        st.markdown("💡 **Tip:** Try different models and adjust temperature/max length for varied responses!")

    elif page == "📁 Files":
        # Files interface
        st.title("📁 Obsidian Files")
        
        obsidian_path = st.session_state.obsidian_path
        
        # Check if path exists
        if not os.path.exists(obsidian_path):
            st.error(f"❌ Directory not found: {obsidian_path}")
            st.info("Please update the path in the sidebar to a valid Obsidian vault directory.")
        else:
            # Get all markdown files
            obsidian_dir = Path(obsidian_path)
            markdown_files = list(obsidian_dir.rglob("*.md"))
            
            if not markdown_files:
                st.warning("⚠️ No markdown files found in the specified directory.")
            else:
                # Create two columns: file list and file content
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📄 Files")
                    
                    # File selection
                    file_names = [f.name for f in markdown_files]
                    file_paths = [str(f) for f in markdown_files]
                    
                    # Create a dictionary for display
                    file_display = {f"{f.name} ({f.parent.name})" if f.parent != obsidian_dir else f.name: str(f) for f in markdown_files}
                    
                    selected_file_display = st.selectbox(
                        "Select a file:",
                        options=list(file_display.keys()),
                        index=0 if st.session_state.selected_file is None else list(file_display.keys()).index(st.session_state.selected_file)
                    )
                    
                    selected_file_path = file_display[selected_file_display]
                    st.session_state.selected_file = selected_file_display
                    
                    # File info
                    if selected_file_path:
                        file_path = Path(selected_file_path)
                        st.info(f"**File:** {file_path.name}\n\n**Path:** {file_path.relative_to(obsidian_dir)}")
                
                with col2:
                    st.subheader("📖 Content")
                    
                    # Display file content
                    if selected_file_path and os.path.exists(selected_file_path):
                        try:
                            with open(selected_file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Display as markdown
                            st.markdown(content)
                            
                            # File statistics
                            st.markdown("---")
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Words", len(content.split()))
                            with col_stats2:
                                st.metric("Characters", len(content))
                            with col_stats3:
                                st.metric("Lines", len(content.splitlines()))
                                
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                    else:
                        st.info("Select a file from the list to view its content.")


if __name__ == "__main__":
    main()