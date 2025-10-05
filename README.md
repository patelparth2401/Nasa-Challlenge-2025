# Scientific Paper RAG System

A Retrieval-Augmented Generation (RAG) system for scientific papers with network visualization capabilities.

## 📁 File Structure

```
.
├── rag_system.py           # Core RAG functionality
├── visualization.py        # Network visualization functions
├── create_embeddings.py    # Script to initialize embeddings
├── app.py                  # Streamlit web application
├── cleaned_data.json       # Your paper data
└── chroma_db/             # ChromaDB storage (created automatically)
```

##  Setup Instructions

### 1. Install Dependencies

```bash
pip install streamlit chromadb ollama keybert spacy plotly networkx
python -m spacy download en_core_web_sm
```

### 2. Install Ollama and Models

Make sure Ollama is running and you have the required models:

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2:1b
```

### 3. Create Embeddings (First Time Only)

Before running the app, you need to create embeddings from your data:

```bash
python create_embeddings.py
```

This will:
- Load all papers from `cleaned_data.json`
- Generate embeddings using Ollama
- Store them in ChromaDB (./chroma_db/)
- Takes a few minutes depending on the number of papers

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

