import json
import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import ollama


class ScientificPaperRAG:
    def __init__(self, json_file_path: str, collection_name: str = "papers"):
        """
        Initialize RAG system for scientific papers
        
        Args:
            json_file_path: Path to your data.json file
            collection_name: Name for ChromaDB collection
        """
        self.json_file_path = json_file_path
        self.collection_name = collection_name
        self.embedding_model = "mxbai-embed-large"
        self.llm_model = "llama3.2:1b"
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.papers = []
        self.collection = None
        
    def load_papers(self):
        """Load papers from JSON file"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)
        print(f"Loaded {len(self.papers)} papers")
        return self.papers
        
    def create_embeddings(self):
        """Create embeddings and store in ChromaDB"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Scientific papers about space research"}
        )
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, paper in enumerate(self.papers):
            # Create searchable text combining all fields
            doc_text = f"""
Title: {paper['Title']}
Authors: {paper['Authors']}
Year: {paper['Year']}
Keywords: {paper['Keywords']}
Abstract: {paper['Abstract']}
            """.strip()
            
            documents.append(doc_text)
            
            # Store metadata for retrieval
            metadatas.append({
                "title": paper['Title'],
                "authors": paper['Authors'],
                "year": str(paper['Year']),
                "keywords": paper['Keywords'],
                "pmid": paper['Links'].get('PMID', ''),
                "doi": paper['Links'].get('DOI', ''),
                "pmc": paper['Links'].get('PMC', '')
            })
            
            ids.append(f"paper_{idx}")
        
        print(f"Creating embeddings for {len(documents)} documents...")
        
        # Generate embeddings using Ollama
        embeddings = []
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"Processing document {i+1}/{len(documents)}...")
            
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=doc
            )
            embeddings.append(response['embedding'])
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully created and stored {len(embeddings)} embeddings!")
        
    def search_papers(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search for relevant papers based on query
        
        Args:
            query: Search query (can be keywords, topics, etc.)
            n_results: Number of results to return
            
        Returns:
            List of relevant papers with metadata
        """
        if self.collection is None:
            self.collection = self.client.get_collection(name=self.collection_name)
        
        # Generate query embedding
        query_embedding = ollama.embeddings(
            model=self.embedding_model,
            prompt=query
        )
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding['embedding']],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def generate_response(self, query: str, n_results: int = 3) -> str:
        """
        Generate LLM response with RAG
        
        Args:
            query: User question
            n_results: Number of papers to retrieve for context
            
        Returns:
            LLM generated response with citations
        """
        # Retrieve relevant papers
        relevant_papers = self.search_papers(query, n_results)
        
        # Build context from retrieved papers
        context = "Here are relevant scientific papers:\n\n"
        for i, paper in enumerate(relevant_papers, 1):
            metadata = paper['metadata']
            context += f"Paper {i}:\n"
            context += f"Title: {metadata['title']}\n"
            context += f"Authors: {metadata['authors']}\n"
            context += f"Year: {metadata['year']}\n"
            context += f"Keywords: {metadata['keywords']}\n"
            if metadata.get('pmid'):
                context += f"PMID: {metadata['pmid']}\n"
            context += f"\nContent:\n{paper['document']}\n"
            context += "-" * 80 + "\n\n"
        
        # Create prompt for LLM
        prompt = f"""Based on the following scientific papers, please answer the user's question.
Include relevant paper titles, authors, and key findings in your response.

{context}

User Question: {query}

Please provide a comprehensive answer citing the relevant papers."""

        # Generate response using Ollama
        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt
        )
        
        return response['response']
    
    def keyword_search(self, keyword: str) -> List[Dict]:
        """
        Simple keyword-based search returning matching papers
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of papers containing the keyword
        """
        matching_papers = []
        for paper in self.papers:
            if (keyword.lower() in paper['Keywords'].lower() or 
                keyword.lower() in paper['Title'].lower() or
                keyword.lower() in paper['Abstract'].lower()):
                matching_papers.append({
                    'title': paper['Title'],
                    'authors': paper['Authors'],
                    'year': paper['Year'],
                    'keywords': paper['Keywords']
                })
        return matching_papers