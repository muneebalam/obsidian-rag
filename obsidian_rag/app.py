import streamlit as st
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
from typing import Optional

# Initialize the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Roberta Chatbot")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Get model predictions
            outputs = model(**inputs)
            
            # Get the predicted answer
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            
            # Get the most likely beginning and end of answer
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            
            # Convert tokens back to text
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)
            
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.markdown(f"Error: {str(e)}")

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap

        streamlit.web.bootstrap.run(__file__, False, [], {})