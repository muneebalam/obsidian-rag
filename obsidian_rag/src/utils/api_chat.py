import os
import streamlit as st
from typing import Dict, Any, Optional
import google.generativeai as genai
import time


def setup_gemini(api_key: str):
    """Setup Gemini API client"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error setting up Gemini: {str(e)}")
        return False


def generate_gemini_response(prompt: str, model_config: Dict[str, Any]) -> str:
    """Generate response using Gemini API"""
    try:
        model_name = model_config.get("model_name", "gemini-2.0-flash-lite")
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 2048)
        
        # Create model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            return "No response generated from Gemini."
            
    except Exception as e:
        return f"Error generating Gemini response: {str(e)}"


def generate_api_response(provider: str, prompt: str, model_config: Dict[str, Any], api_key: str) -> str:
    """Generate response using the specified API provider"""
    
    if provider == "gemini":
        if not setup_gemini(api_key):
            return "Error: Failed to setup Gemini API"
        return generate_gemini_response(prompt, model_config)
    else:
        return f"Error: Unknown API provider '{provider}'"


def validate_api_connection(provider: str, api_key: str) -> bool:
    """Validate API connection for the specified provider"""
    try:
        if provider == "gemini":
            return setup_gemini(api_key)
        else:
            return False
    except Exception:
        return False 