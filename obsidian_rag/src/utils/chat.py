import torch
from .api_chat import generate_api_response
from .config import get_api_key, load_config
from .models import MODELS_CONFIG


def generate_response(prompt: str, tokenizer, model, model_config: dict) -> str:
    """Generate response using the selected model (local or API)"""
    
    # Check if this is an API model
    if model_config.get("type") == "api":
        return generate_api_response_wrapper(prompt, model_config)
    else:
        # Local model generation
        return generate_local_response(prompt, tokenizer, model, model_config)


def generate_api_response_wrapper(prompt: str, model_config: dict) -> str:
    """Wrapper for API-based model responses"""
    try:
        provider = model_config.get("provider")
        config = load_config()
        api_key = get_api_key(provider, config)
        
        if not api_key:
            return f"Error: No API key found for {provider}. Please set the {provider.upper()}_API_KEY environment variable or add it to config.yaml"
        
        return generate_api_response(provider, prompt, model_config, api_key)
        
    except Exception as e:
        return f"Error generating API response: {str(e)}"


def generate_local_response(prompt: str, tokenizer, model, model_config: dict) -> str:
    """Generate response using local model"""
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