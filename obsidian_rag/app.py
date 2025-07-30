import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
import glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple

# List of available models
AVAILABLE_MODELS = {
    "Roberta Base (SQuAD2)": "deepset/roberta-base-squad2",
    "DistilBERT (SQuAD2)": "deepset/distilbert-base-uncased-distilled-squad",
    "BERT (SQuAD2)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "MiniLM (SQuAD2)": "deepset/minilm-uncased-squad2"
}

# Querying technique selection
QUERYING_TECHNIQUES = {
    "Step Back": "step_back",
    "Multi Query": "multi_query",
    "Drill Down": "drill_down"
}

# Initialize models
@st.cache_resource
def load_models():
    """Load both the question-answering and embedding models"""
    try:
        # Load question-answering model
        qa_model_name = "deepset/roberta-base-squad2"
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return qa_tokenizer, qa_model, embedding_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None

qa_tokenizer, qa_model, embedding_model = load_models()

if not qa_model:
    st.error("Failed to load the models. Please try again.")
    st.stop()

# Streamlit UI
st.title("Question Answering Chatbot")

# Obsidian Database Directory Selection
def load_obsidian_notes(directory):
    """Load all markdown files from the Obsidian database and create embeddings"""
    notes = []
    try:
        # Find all markdown files in the directory and subdirectories
        markdown_files = glob.glob(os.path.join(directory, '**/*.md'), recursive=True)
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract just the filename without extension for the note title
                    note_title = os.path.splitext(os.path.basename(file_path))[0]
                    notes.append({
                        'title': note_title,
                        'content': content,
                        'path': file_path
                    })
            except Exception as e:
                st.error(f"Error reading file {file_path}: {str(e)}")
                continue
        
        # Create embeddings for all notes
        note_texts = [note['content'] for note in notes]
        note_embeddings = embedding_model.encode(note_texts)
        
        # Add embeddings to each note
        for note, embedding in zip(notes, note_embeddings):
            note['embedding'] = embedding
        
        return notes
    except Exception as e:
        st.error(f"Error loading Obsidian notes: {str(e)}")
        return []

# Default to DS docs directory
DEFAULT_OBSIDIAN_DIR = "/Users/muneebalam/Documents/DS docs"

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0
    )
    
    # Querying technique selection
    selected_technique = st.selectbox(
        "Querying Technique",
        options=list(QUERYING_TECHNIQUES.keys()),
        index=0
    )
    
    # Obsidian database directory
    obsidian_dir = st.text_input(
        "Obsidian Database Directory",
        value=DEFAULT_OBSIDIAN_DIR,
        help="Path to your Obsidian database directory containing markdown files"
    )
    
    # Load notes when directory changes
    if obsidian_dir and obsidian_dir != st.session_state.get('obsidian_dir', ''):
        st.session_state.obsidian_dir = obsidian_dir
        st.session_state.notes = load_obsidian_notes(obsidian_dir)

# Load selected model
model_name = AVAILABLE_MODELS[selected_model]
tokenizer, model = load_model(model_name)

if not model:
    st.error("Failed to load the selected model. Please try another model.")
    st.stop()

# Display loaded notes
if st.session_state.get('notes'):
    st.subheader("Available Notes")
    for note in st.session_state.notes:
        with st.expander(note['title']):
            st.markdown(note['content'])

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(AVAILABLE_MODELS.keys()),
    index=0
)

selected_technique = st.sidebar.selectbox(
    "Querying Technique",
    options=list(QUERYING_TECHNIQUES.keys()),
    index=0
)

def get_answer(question, context, technique):
    """Generate answer using the selected querying technique"""
    if technique == "step_back":
        # Step back technique: Break down the question into simpler parts
        simplified_question = question
        if "what" in question.lower():
            simplified_question = question.replace("what", "which")
        elif "how" in question.lower():
            simplified_question = question.replace("how", "what")
        
        # Tokenize and get answer
        inputs = tokenizer(simplified_question, context, return_tensors="pt")
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )
        return answer
    
    elif technique == "multi_query":
        # Multi query technique: Generate multiple variations of the question
        variations = [
            question,
            question.replace("what", "which"),
            question.replace("how", "what"),
            question.replace("why", "what")
        ]
        
        answers = []
        for var in variations:
            try:
                inputs = tokenizer(var, context, return_tensors="pt")
                outputs = model(**inputs)
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
                )
                answers.append(answer)
            except:
                continue
        
        # Return the most common answer
        if answers:
            return max(set(answers), key=answers.count)
        return "No answer found"
    
    elif technique == "drill_down":
        # Drill down technique: Get initial answer and refine it
        inputs = tokenizer(question, context, return_tensors="pt")
        outputs = model(**inputs)
        
        # Get initial answer
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        initial_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )
        
        # Refine the answer by focusing on the relevant part of the context
        refined_context = context[answer_start:answer_end]
        refined_inputs = tokenizer(question, refined_context, return_tensors="pt")
        refined_outputs = model(**refined_inputs)
        
        refined_start = torch.argmax(refined_outputs.start_logits)
        refined_end = torch.argmax(refined_outputs.end_logits) + 1
        refined_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(refined_inputs.input_ids[0][refined_start:refined_end])
        )
        
        return refined_answer
    
    # Default to regular question answering
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )
    return answer

# Load selected model
model_name = AVAILABLE_MODELS[selected_model]
tokenizer, model = load_model(model_name)

if not model:
    st.error("Failed to load the selected model. Please try another model.")
    st.stop()

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
            # Split the input into question and context
            # For now, we'll use a default context (you might want to customize this)
            question = prompt
            context = """Robert is a friendly AI assistant designed to help users with their questions.
            He is knowledgeable about many topics and always tries to provide accurate and helpful answers.
            Robert was created by the Hugging Face team and is based on the RoBERTa model architecture.
            The model has been fine-tuned on the SQuAD 2.0 dataset for question answering tasks.
            """
            
            # Get answer using the selected querying technique
            # Find the most relevant note using embeddings
            if st.session_state.get('notes'):
                # Get embeddings for the question
                question_embedding = embedding_model.encode(question)
                
                # Calculate cosine similarity with all notes
                similarities = []
                for note in st.session_state.notes:
                    similarity = np.dot(question_embedding, note['embedding']) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(note['embedding'])
                    )
                    similarities.append((note, similarity))
                
                # Sort by similarity and get top 3 most relevant notes
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_notes = [note for note, _ in similarities[:3]]
                
                # Combine content from top notes
                context = "\n\n".join([note['content'] for note in top_notes])
                
                # Display the relevant notes in the chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Using context from: {', '.join(note['title'] for note in top_notes)}"
                })
            else:
                context = """Robert is a friendly AI assistant designed to help users with their questions.
                He is knowledgeable about many topics and always tries to provide accurate and helpful answers.
                Robert was created by the Hugging Face team and is based on the RoBERTa model architecture.
                The model has been fine-tuned on the SQuAD 2.0 dataset for question answering tasks.
                """
            
            # Get answer using the selected querying technique
            answer = get_answer(question, context, QUERYING_TECHNIQUES[selected_technique])
            
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