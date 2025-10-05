"""
Script to create embeddings from cleaned_data.json
Run this once to initialize the database
"""

from rag_system import ScientificPaperRAG

def main():
    # Initialize RAG system
    rag = ScientificPaperRAG("cleaned_data.json")
    
    # Load papers
    rag.load_papers()
    
    # Create embeddings (this will take some time)
    print("\nCreating embeddings... This may take a few minutes.")
    rag.create_embeddings()
    
    print("\n✓ Embeddings created successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()