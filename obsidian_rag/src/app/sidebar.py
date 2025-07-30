import streamlit as st
from src.utils import MODELS_CONFIG
from src.app.model_settings import load_model_and_tokenizer, get_available_models, validate_model_access
from src.utils import EMBEDDING_MODELS, process_obsidian_files, embed_documents, save_vector_db, load_vector_db, load_config
import torch
import gc
import os
from dotenv import load_dotenv
load_dotenv()

def sidebar_page():
    """Sidebar configuration and model management"""
    try:
        # Load configuration
        config = load_config()
        
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

        # Get available models based on API key availability
        available_models = get_available_models(config)
        
        if not available_models:
            st.sidebar.error("❌ No models available. Please check your API keys or internet connection.")
            return "💬 Chat", None, None

        # Model selection dropdown
        selected_model_name = st.sidebar.selectbox(
            "Choose a Model:",
            available_models,
            index=0
        )

        # Get selected model config
        selected_config = MODELS_CONFIG[selected_model_name].copy()

        # Show model type and status
        model_type = selected_config.get("type", "local")
        if model_type == "api":
            provider = selected_config.get("provider", "unknown")
            api_key = os.environ[f"{provider.upper()}_API_KEY"] #config.get("api_keys", {}).get(provider, "")]
            if api_key:
                st.sidebar.success(f"✅ {provider.upper()} API key available")
            else:
                st.sidebar.error(f"❌ No {provider.upper()} API key found")
        else:
            st.sidebar.info("🏠 Local model")

        # Model parameters
        st.sidebar.subheader("Generation Parameters")
        
        if model_type == "api":
            # API model parameters
            temperature = st.sidebar.slider("Temperature", 0.1, 1.0, selected_config["temperature"], 0.1)
            max_tokens = st.sidebar.slider("Max Tokens", 100, 4000, selected_config["max_tokens"], 100)
            
            # Update config with sidebar values
            selected_config["temperature"] = temperature
            selected_config["max_tokens"] = max_tokens
        else:
            # Local model parameters
            temperature = st.sidebar.slider("Temperature", 0.1, 1.0, selected_config["temperature"], 0.1)
            max_length = st.sidebar.slider("Max Length", 50, 200, selected_config["max_length"], 10)
            
            # Update config with sidebar values
            selected_config["temperature"] = temperature
            selected_config["max_length"] = max_length

        # Load model if changed (only for local models)
        if model_type == "local":
            if (st.session_state.current_model != selected_config["model_name"] or 
                st.session_state.tokenizer is None or 
                st.session_state.model is None):
                
                with st.spinner(f"Loading {selected_model_name}..."):
                    try:
                        # Clear previous model from memory
                        if st.session_state.model is not None:
                            del st.session_state.model
                            del st.session_state.tokenizer
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Load new model
                        tokenizer, model = load_model_and_tokenizer(selected_config["model_name"])
                        
                        if tokenizer is not None and model is not None:
                            st.session_state.tokenizer = tokenizer
                            st.session_state.model = model
                            st.session_state.current_model = selected_config["model_name"]
                            st.sidebar.success(f"✅ {selected_model_name} loaded successfully!")
                        else:
                            st.sidebar.error(f"❌ Failed to load {selected_model_name}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error loading model: {str(e)}")

        # Display model info
        if model_type == "local" and st.session_state.current_model:
            st.sidebar.subheader("Current Model")
            st.sidebar.info(f"**Model:** {selected_model_name}\n\n**Path:** {st.session_state.current_model}")
        elif model_type == "api":
            st.sidebar.subheader("API Model")
            st.sidebar.info(f"**Model:** {selected_model_name}\n\n**Provider:** {selected_config['provider'].upper()}\n\n**Type:** API-based")

        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # API Key Status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 API Key Status")
        
        api_status = {}
        for provider in ["gemini"]:
            api_key = config.get("api_keys", {}).get(provider, "")
            if api_key and api_key.strip():
                api_status[provider] = "✅ Available"
            else:
                api_status[provider] = "❌ Missing"
        
        for provider, status in api_status.items():
            st.sidebar.text(f"{provider.upper()}: {status}")

        # Embedding section
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Vector Database")
        
        # Embedding model selection
        selected_embedding_model = st.sidebar.selectbox(
            "Embedding Model:",
            list(EMBEDDING_MODELS.keys()),
            index=0,
            help="Select an embedding model for vector database creation"
        )
        
        # Show embedding model info
        embedding_info = EMBEDDING_MODELS[selected_embedding_model]
        st.sidebar.info(f"**Model:** {selected_embedding_model}\n\n**Dimensions:** {embedding_info['dimensions']}\n\n**Description:** {embedding_info['description']}")
        
        # Chunking parameters
        st.sidebar.subheader("📄 Chunking Settings")
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100, help="Size of text chunks in characters")
        overlap = st.sidebar.slider("Overlap", 50, 500, 200, 50, help="Overlap between chunks in characters")
        
        # Embed database button
        if st.sidebar.button("🚀 Embed Obsidian Database", type="primary"):
            if obsidian_path and os.path.exists(obsidian_path):
                try:
                    with st.spinner("Processing Obsidian files..."):
                        # Process files
                        documents = process_obsidian_files(obsidian_path, chunk_size, overlap)
                        
                        if documents:
                            st.sidebar.success(f"✅ Processed {len(documents)} document chunks")
                            
                            # Embed documents
                            with st.spinner("Creating embeddings..."):
                                vector_db = embed_documents(documents, selected_embedding_model)
                                
                                # Save vector database
                                if save_vector_db(vector_db):
                                    st.session_state.vector_db = vector_db
                                    st.sidebar.success("✅ Vector database created and saved!")
                                else:
                                    st.sidebar.error("❌ Failed to save vector database")
                        else:
                            st.sidebar.warning("⚠️ No documents found to process")
                            
                except Exception as e:
                    st.sidebar.error(f"❌ Error embedding database: {str(e)}")
            else:
                st.sidebar.error("❌ Please enter a valid Obsidian directory path")
        
        # Load existing vector database
        if st.sidebar.button("📂 Load Existing Database"):
            vector_db = load_vector_db()
            if vector_db:
                st.session_state.vector_db = vector_db
                st.sidebar.success(f"✅ Loaded vector database with {vector_db['total_documents']} documents")
            else:
                st.sidebar.warning("⚠️ No existing vector database found")
        
        # Show vector database info
        if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
            db_info = st.session_state.vector_db
            st.sidebar.subheader("📊 Database Info")
            st.sidebar.info(f"**Documents:** {db_info['total_documents']}\n\n**Model:** {db_info['model_name']}\n\n**Created:** {db_info['created_at'][:10]}")

        # Page selection
        st.sidebar.markdown("---")
        page = st.sidebar.radio("📄 Pages", ["💬 Chat", "📁 Files"])
        
        # Store obsidian path in session state
        st.session_state.obsidian_path = obsidian_path
        
        return page, selected_model_name, selected_config
        
    except Exception as e:
        st.sidebar.error(f"Sidebar error: {str(e)}")
        # Return default values if there's an error
        return "💬 Chat", list(MODELS_CONFIG.keys())[0], MODELS_CONFIG[list(MODELS_CONFIG.keys())[0]]