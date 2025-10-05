import json
import networkx as nx
import plotly.graph_objects as go
from keybert import KeyBERT
import spacy


class NetworkVisualization:
    def __init__(self, json_file_path: str):
        """
        Initialize visualization system
        
        Args:
            json_file_path: Path to cleaned_data.json
        """
        self.json_file_path = json_file_path
        self.data = []
        self.kw_model = KeyBERT('all-MiniLM-L6-v2')
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def load_data(self):
        """Load papers from JSON file"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data
    
    def extract_keywords_from_text(self, text: str, top_n: int = 20):
        """
        Extract keywords from text using KeyBERT
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        keywords = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=top_n
        )
        return keywords
    
    def extract_entities(self, text: str):
        """
        Extract named entities using spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with authors and topics
        """
        if self.nlp is None:
            return {'authors': [], 'topics': []}
        
        doc = self.nlp(text)
        authors = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        topics = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "NORP", "GPE")]
        
        return {'authors': authors, 'topics': topics}
    
    def create_author_keyword_network(self):
        """
        Create network visualization of authors and keywords
        
        Returns:
            Plotly figure object
        """
        if not self.data:
            self.load_data()
        
        # Initialize the graph
        G = nx.Graph()
        
        # Build graph edges (Author ↔ Keyword)
        for paper in self.data:
            authors = [a.strip() for a in paper["Authors"].split(",") if a.strip()]
            keywords = [k.strip() for k in paper["Keywords"].split(",") if k.strip()]
            
            for author in authors:
                G.add_node(author, type='author')
            for kw in keywords:
                G.add_node(kw, type='keyword')
            
            for author in authors:
                for kw in keywords:
                    if G.has_edge(author, kw):
                        G[author][kw]['weight'] = G[author][kw].get('weight', 1) + 1
                    else:
                        G.add_edge(author, kw, weight=1)
        
        # Layout positions for visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Separate nodes by type
        author_x, author_y, author_names = [], [], []
        keyword_x, keyword_y, keyword_names = [], [], []
        
        for node, attrs in G.nodes(data=True):
            x, y = pos[node]
            if attrs['type'] == 'author':
                author_x.append(x)
                author_y.append(y)
                author_names.append(node)
            else:
                keyword_x.append(x)
                keyword_y.append(y)
                keyword_names.append(node)
        
        # Build Plotly edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add authors
        fig.add_trace(go.Scatter(
            x=author_x, y=author_y,
            mode='markers+text',
            text=author_names,
            textposition='top center',
            textfont=dict(size=8),
            hovertext=['Author: ' + name for name in author_names],
            hoverinfo='text',
            marker=dict(size=12, color='blue', opacity=0.7),
            name='Authors'
        ))
        
        # Add keywords
        fig.add_trace(go.Scatter(
            x=keyword_x, y=keyword_y,
            mode='markers+text',
            text=keyword_names,
            textposition='top center',
            textfont=dict(size=8),
            hovertext=['Keyword: ' + name for name in keyword_names],
            hoverinfo='text',
            marker=dict(size=10, color='orange', opacity=0.7),
            name='Keywords'
        ))
        
        fig.update_layout(
            title="Author–Keyword Relationship Network",
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        return fig
    
    def create_keyword_cooccurrence_network(self, min_cooccurrence: int = 2):
        """
        Create network showing keyword co-occurrences
        
        Args:
            min_cooccurrence: Minimum number of co-occurrences to show edge
            
        Returns:
            Plotly figure object
        """
        if not self.data:
            self.load_data()
        
        G = nx.Graph()
        
        # Build keyword co-occurrence graph
        for paper in self.data:
            keywords = [k.strip() for k in paper["Keywords"].split(",") if k.strip()]
            
            # Add all keyword pairs
            for i, kw1 in enumerate(keywords):
                G.add_node(kw1)
                for kw2 in keywords[i+1:]:
                    if G.has_edge(kw1, kw2):
                        G[kw1][kw2]['weight'] += 1
                    else:
                        G.add_edge(kw1, kw2, weight=1)
        
        # Filter edges by minimum co-occurrence
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_cooccurrence]
        G.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if len(G.nodes()) == 0:
            # Return empty figure if no connections
            fig = go.Figure()
            fig.add_annotation(
                text="No keyword co-occurrences found with current threshold",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Node sizes based on degree
        node_degrees = dict(G.degree())
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        node_x, node_y, node_names, node_sizes = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            node_sizes.append(10 + (node_degrees[node] / max_degree) * 20)
        
        # Edges with weights
        edge_x, edge_y, edge_weights = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_weights.append(edge[2]['weight'])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_names,
            textposition='top center',
            textfont=dict(size=8),
            hovertext=[f'{name}<br>Connections: {node_degrees[name]}' for name in node_names],
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color='green',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Keyword Co-occurrence Network",
            title_x=0.5,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        return fig