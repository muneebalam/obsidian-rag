import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional
import gc
import os
from pathlib import Path
from src.utils import MODELS_CONFIG
from src.app.model_settings import load_model_and_tokenizer
from src.app.sidebar import sidebar_page
from src.utils import generate_response, search_similar_documents
from dotenv import load_dotenv
load_dotenv()


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
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = False


def create_rag_prompt(user_query: str, relevant_docs: list, max_context_length: int = 2000) -> str:
    """Create a RAG-enhanced prompt with relevant context"""
    if not relevant_docs:
        return user_query
    
    # Build context from relevant documents
    context_parts = []
    current_length = 0
    
    for doc_result in relevant_docs:
        doc = doc_result["document"]
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Format the context with metadata
        context_entry = f"From file '{metadata['filename']}' (folder: {metadata['folder']}):\n{content}\n"
        
        # Check if adding this would exceed the limit
        if current_length + len(context_entry) > max_context_length:
            break
            
        context_parts.append(context_entry)
        current_length += len(context_entry)
    
    if context_parts:
        context = "\n".join(context_parts)
        rag_prompt = f"""Based on the following information from my knowledge base, please answer the user's question:

{context}

User Question: {user_query}

Please provide a helpful answer based on the information above. If the information doesn't fully address the question, you can provide additional insights based on your general knowledge."""
    else:
        rag_prompt = user_query
    
    return rag_prompt


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Get sidebar configuration
    page, selected_model_name, selected_config = sidebar_page()
    
    # Check if we have valid model configuration
    if selected_model_name is None or selected_config is None:
        st.error("❌ No valid model configuration available. Please check your API keys or model settings.")
        return

    # Main content area
    if page == "💬 Chat":
        # Chat interface
        st.title("🤖 AI Chatbot")
        st.markdown(f"**Current Model:** {selected_model_name}")
        
        # Show model type
        model_type = selected_config.get("type", "local")
        if model_type == "api":
            provider = selected_config.get("provider", "unknown")
            st.info(f"🌐 Using {provider.upper()} API model")
        else:
            st.info("🏠 Using local model")
        
        # RAG controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            rag_enabled = st.checkbox(
                "🔍 Enable RAG", 
                value=st.session_state.rag_enabled,
                help="Enable retrieval-augmented generation using your Obsidian database"
            )
            st.session_state.rag_enabled = rag_enabled
        
        with col2:
            if rag_enabled:
                top_k = st.slider("Top K Results", 1, 10, 3, help="Number of relevant documents to retrieve")
            else:
                top_k = 3
        
        with col3:
            if rag_enabled:
                max_context = st.slider("Max Context (chars)", 1000, 4000, 2000, 100, help="Maximum context length in characters")
            else:
                max_context = 2000
        
        # Show RAG status
        if rag_enabled:
            if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
                db_info = st.session_state.vector_db
                st.success(f"✅ RAG Enabled - Using {db_info['total_documents']} documents from {db_info['model_name']}")
            else:
                st.warning("⚠️ RAG Enabled but no vector database loaded. Please create or load a vector database in the sidebar.")
        else:
            st.info("ℹ️ RAG Disabled - Using general knowledge only")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show RAG context for assistant messages if available
                if message["role"] == "assistant" and "rag_context" in message:
                    with st.expander("🔍 RAG Context Used"):
                        for i, doc_result in enumerate(message["rag_context"]):
                            doc = doc_result["document"]
                            metadata = doc["metadata"]
                            st.markdown(f"**{i+1}. {metadata['filename']}** (similarity: {doc_result['similarity']:.3f})")
                            st.markdown(f"*Folder: {metadata['folder']}*")
                            st.markdown(f"```\n{doc['content'][:200]}...\n```")

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                # Check if we have the required components for the model type
                if model_type == "local":
                    # Local model requires tokenizer and model
                    if st.session_state.tokenizer is not None and st.session_state.model is not None:
                        can_generate = True
                    else:
                        can_generate = False
                        error_msg = "Local model not loaded. Please select a model from the sidebar."
                else:
                    # API model doesn't require local components
                    can_generate = True
                
                if can_generate:
                    with st.spinner("Generating response..."):
                        # RAG retrieval if enabled and database available
                        rag_context = []
                        final_prompt = prompt
                        
                        if (rag_enabled and 
                            hasattr(st.session_state, 'vector_db') and 
                            st.session_state.vector_db):
                            
                            # Search for relevant documents
                            with st.spinner("Searching knowledge base..."):
                                rag_context = search_similar_documents(
                                    prompt, 
                                    st.session_state.vector_db, 
                                    top_k
                                )
                            
                            # Create RAG-enhanced prompt
                            final_prompt = create_rag_prompt(prompt, rag_context, max_context)
                        
                        # Generate response
                        response = generate_response(
                            final_prompt, 
                            st.session_state.tokenizer, 
                            st.session_state.model, 
                            selected_config
                        )
                        
                        # Add assistant response to chat history with RAG context
                        message_data = {"role": "assistant", "content": response}
                        if rag_context:
                            message_data["rag_context"] = rag_context
                        
                        st.session_state.messages.append(message_data)
                        st.markdown(response)
                        
                        # Show RAG info if context was used
                        if rag_context:
                            st.info(f"🔍 Retrieved {len(rag_context)} relevant documents from your knowledge base")
                else:
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)

        # Footer
        st.markdown("---")
        if rag_enabled:
            st.markdown("💡 **Tip:** RAG is enabled! The AI will use your Obsidian knowledge base to provide more relevant answers.")
        else:
            st.markdown("💡 **Tip:** Enable RAG to use your Obsidian knowledge base for more relevant responses!")

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