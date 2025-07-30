import os
import streamlit as st
from typing import Dict, Any, Optional
from google import genai
import time



def generate_gemini_response(prompt: str, model_config: Dict[str, Any]) -> str:
    """Generate response using Gemini API"""
    try:
        model_name = model_config.get("model_name", "gemini-2.0-flash-lite")
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 2048)
        
        # Create model
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Generate response
        response = client.models.generate_content(model=model_name, contents=prompt)
        
        if response.text:
            return response.text
        else:
            return "No response generated from Gemini."
            
    except Exception as e:
        return f"Error generating Gemini response: {str(e)}"


def generate_api_response(provider: str, prompt: str, model_config: Dict[str, Any], api_key: str) -> str:
    """Generate response using the specified API provider"""
    
    return generate_gemini_response(prompt, model_config)

