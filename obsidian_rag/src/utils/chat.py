import torch

def generate_response(prompt: str, tokenizer, model, model_config: dict) -> str:
    """Generate response using the selected model"""
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