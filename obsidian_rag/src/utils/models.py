# Available models configuration
MODELS_CONFIG = {
    "GPT-2 (Medium)": {
        "model_name": "gpt2-medium",
        "max_length": 100,
        "temperature": 0.7,
        "type": "local"
    },
    "BLOOM (560M)": {
        "model_name": "bigscience/bloom-560m",
        "max_length": 100,
        "temperature": 0.7,
        "type": "local"
    },
    "DialoGPT (Medium)": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 100,
        "temperature": 0.7,
        "type": "local"
    },
    "TinyLlama": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_length": 150,
        "temperature": 0.7,
        "type": "local"
    },
    # API Models
    "Gemini 2.0": {
        "model_name": "gemini-2.0-flash-lite",
        "max_tokens": 2048,
        "temperature": 0.7,
        "type": "api",
        "provider": "gemini"
    },
}
