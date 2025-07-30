# Obsidian notes chatbot

This WIP repo performs retrieval-augmented generation on an Obsidian database.

It is also meant as an example of different RAG techniques. 

How it works (high level):

- Chunk and embed text using RoBERTa from HuggingFace
- Embed user question using same model
- Perform a lookup and retrieve similar chunks
- Prompt LLM and return answer

# Setup

1. Install `uv` (see [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods))
2. Install dependencies: `uv sync`

# Running

1. Run `uv run streamlit run app.py`

# To do

- Configurable embedding model and LLM reasoning model
- Querying techniques: step back, multi query, drill down
- Indexing techniques: multi-representation indexing, different chunking techniques, RAPTOR, ColBERT, knowledge graph
- Retrieval techniques: 
- Agentic chatbot (including agentic workflows - logical and semantic routing, query structuring, which DB to use)
- Observability
- Fine tuning
- RLHF
- Graph DB
- Knowledge graph
