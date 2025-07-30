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
        # Validate and sanitize input
        if not prompt or not prompt.strip():
            return "Please provide a valid input prompt."
        
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
        
        # Validate input length
        if inputs.shape[1] > 1024:  # Limit input length for GPT-2
            inputs = inputs[:, -1024:]  # Take last 1024 tokens
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Generate response with safer parameters
        with torch.no_grad():
            try:
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + model_config["max_length"],
                    temperature=max(0.1, min(1.0, model_config["temperature"])),  # Clamp temperature
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    # Add safety parameters
                    top_k=50,  # Limit vocabulary to top 50 tokens
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.1,  # Prevent repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    # Error handling
                    return_dict_in_generate=True,
                    output_scores=False,  # Don't return scores to avoid probability issues
                )
                
                # Extract the generated tokens
                if hasattr(outputs, 'sequences'):
                    generated_tokens = outputs.sequences[0]
                else:
                    generated_tokens = outputs[0]
                
            except RuntimeError as e:
                if "probability tensor" in str(e).lower():
                    # Fallback to greedy decoding if sampling fails
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + model_config["max_length"],
                        do_sample=False,  # Use greedy decoding
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        attention_mask=torch.ones_like(inputs),
                    )
                    generated_tokens = outputs[0]
                else:
                    raise e
        
        # Decode response
        if "dialo" in model_config["model_name"].lower():
            response = tokenizer.decode(generated_tokens[inputs.shape[-1]:], skip_special_tokens=True)
        elif "tinyllama" in model_config["model_name"].lower():
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Extract assistant response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].split("</s>")[0].strip()
        else:
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
        
        # Clean up response
        response = response.strip()
        if not response:
            return "I'm not sure how to respond to that."
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"