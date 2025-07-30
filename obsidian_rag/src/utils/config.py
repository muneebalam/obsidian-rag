import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            st.warning(f"Configuration file {config_path} not found. Using defaults.")
            return get_default_config()
        
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Substitute environment variables
        config_content = substitute_env_vars(config_content)
        
        # Parse YAML
        config = yaml.safe_load(config_content)
        
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return get_default_config()


def substitute_env_vars(content: str) -> str:
    """Substitute environment variables in string content"""
    def replace_env_var(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))
    
    # Replace ${ENV_VAR} patterns
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_env_var, content)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration if config file is not available"""
    return {
        "api_keys": {
            "gemini": os.getenv("GEMINI_API_KEY", "")
        },
        "models": {
            "gemini": {
                "model_name": "gemini-2.0-flash-lite",
                "max_tokens": 2048,
                "temperature": 0.7
            }
        },
        "app": {
            "default_chat_model": "GPT-2 (Medium)",
            "default_embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_context_length": 2000,
            "top_k_results": 3
        }
    }


def get_api_key(provider: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Get API key for a specific provider"""
    if config is None:
        config = load_config()
    
    api_key = config.get("api_keys", {}).get(provider)
    
    if not api_key:
        # Try environment variable directly
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var_name)
    
    return api_key


def get_model_config(provider: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get model configuration for a specific provider"""
    if config is None:
        config = load_config()
    
    return config.get("models", {}).get(provider, {})


def validate_api_keys(config: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that required API keys are available"""
    validation = {}
    
    for provider in ["gemini"]:
        api_key = get_api_key(provider, config)
        validation[provider] = bool(api_key and api_key.strip())
    
    return validation 