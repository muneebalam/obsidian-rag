# Obsidian notes chatbot

This WIP repo performs retrieval-augmented generation on an Obsidian database.

It is also meant as an example of different RAG techniques. 

How it works (high level):

- Chunk and embed text using RoBERTa from HuggingFace
- Embed user question using same model
- Perform a lookup and retrieve similar chunks
- Prompt LLM and return answer

# To do

- Streamlit interface
- Configurable embedding model and LLM reasoning model
- Querying techniques: step back, multi query, drill down
- Indexing techniques: multi-representation indexing, different chunking techniques, RAPTOR, ColBERT, knowledge graph
- Retrieval techniques: 
- Agentic chatbot (including agentic workflows - logical and semantic routing, query structuring)
- Observability
- Fine tuning
- RLHF
