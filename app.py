import streamlit as st
from rag_system import ScientificPaperRAG
from visualization import NetworkVisualization
import json

# Page config
st.set_page_config(
    page_title="Scientific Paper RAG System",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = ScientificPaperRAG("cleaned_data.json")
    st.session_state.rag.load_papers()

if 'viz' not in st.session_state:
    st.session_state.viz = NetworkVisualization("cleaned_data.json")
    st.session_state.viz.load_data()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("🔬 Scientific Paper RAG System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", ["RAG Query", "Network Visualization", "Paper Browser"])
    
    st.markdown("---")
    st.header("About")
    st.info(f"📚 Total Papers: {len(st.session_state.rag.papers)}")
    st.markdown("""
    This system uses:
    - **RAG** (Retrieval Augmented Generation)
    - **ChromaDB** for embeddings
    - **Ollama** for LLM inference
    - **Network Visualization** for insights
    """)

# Main content
if page == "RAG Query":
    st.header("Ask Questions About the Papers")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Enter your question:", placeholder="What are the effects of microgravity on bone loss?")
        n_results = st.slider("Number of papers to retrieve:", 1, 10, 3)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching papers and generating response..."):
                    try:
                        # Get relevant papers
                        relevant_papers = st.session_state.rag.search_papers(query, n_results)
                        
                        # Generate response
                        response = st.session_state.rag.generate_response(query, n_results)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': query,
                            'response': response,
                            'papers': relevant_papers
                        })
                        
                        st.success("Response generated!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("Quick Stats")
        if st.session_state.chat_history:
            st.metric("Total Queries", len(st.session_state.chat_history))
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Query History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query']}", expanded=(i==0)):
                st.markdown("**Response:**")
                st.write(chat['response'])
                
                st.markdown("**Relevant Papers:**")
                for j, paper in enumerate(chat['papers'], 1):
                    metadata = paper['metadata']
                    distance_str = f"{paper['distance']:.4f}" if paper['distance'] is not None else 'N/A'
                    st.markdown(f"""
                    **{j}. {metadata['title']}**
                    - Authors: {metadata['authors']}
                    - Year: {metadata['year']}
                    - Keywords: {metadata['keywords']}
                    - Distance: {distance_str}
                    """)

elif page == "Network Visualization":
    st.header("Network Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Author-Keyword Network", "Keyword Co-occurrence Network"]
    )
    
    if viz_type == "Author-Keyword Network":
        st.markdown("""
        This network shows the relationships between authors and keywords.
        - **Blue nodes**: Authors
        - **Orange nodes**: Keywords
        - **Lines**: Connections between authors and their research keywords
        """)
        
        with st.spinner("Generating network visualization..."):
            fig = st.session_state.viz.create_author_keyword_network()
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Keyword Co-occurrence Network
        st.markdown("""
        This network shows how keywords appear together in papers.
        - **Node size**: Number of connections (more connections = larger node)
        - **Lines**: Keywords that appear together in papers
        """)
        
        min_cooccurrence = st.slider("Minimum co-occurrences:", 1, 5, 2)
        
        with st.spinner("Generating co-occurrence network..."):
            fig = st.session_state.viz.create_keyword_cooccurrence_network(min_cooccurrence)
            st.plotly_chart(fig, use_container_width=True)

else:  # Paper Browser
    st.header("Browse Papers")
    
    # Search options
    search_type = st.radio("Search Type", ["Keyword Search", "Browse All"])
    
    if search_type == "Keyword Search":
        keyword = st.text_input("Enter keyword:", placeholder="microgravity")
        
        if keyword:
            matching_papers = st.session_state.rag.keyword_search(keyword)
            
            st.subheader(f"Found {len(matching_papers)} papers")
            
            for i, paper in enumerate(matching_papers, 1):
                with st.expander(f"{i}. {paper['title']}"):
                    st.markdown(f"**Authors:** {paper['authors']}")
                    st.markdown(f"**Year:** {paper['year']}")
                    st.markdown(f"**Keywords:** {paper['keywords']}")
    
    else:  # Browse All
        st.subheader(f"All Papers ({len(st.session_state.rag.papers)})")
        
        # Add sorting
        sort_by = st.selectbox("Sort by:", ["Year (Newest)", "Year (Oldest)", "Title"])
        
        papers = st.session_state.rag.papers.copy()
        if sort_by == "Year (Newest)":
            papers.sort(key=lambda x: x['Year'], reverse=True)
        elif sort_by == "Year (Oldest)":
            papers.sort(key=lambda x: x['Year'])
        else:
            papers.sort(key=lambda x: x['Title'])
        
        # Pagination
        papers_per_page = 10
        total_pages = (len(papers) + papers_per_page - 1) // papers_per_page
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_num - 1) * papers_per_page
        end_idx = min(start_idx + papers_per_page, len(papers))
        
        for i, paper in enumerate(papers[start_idx:end_idx], start_idx + 1):
            with st.expander(f"{i}. {paper['Title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authors:** {paper['Authors']}")
                    st.markdown(f"**Year:** {paper['Year']}")
                    st.markdown(f"**Keywords:** {paper['Keywords']}")
                    st.markdown(f"**Abstract:** {paper['Abstract']}")
                
                with col2:
                    st.markdown("**Links:**")
                    if paper['Links'].get('PMID'):
                        st.markdown(f"[PMID: {paper['Links']['PMID']}](https://pubmed.ncbi.nlm.nih.gov/{paper['Links']['PMID']})")
                    if paper['Links'].get('DOI'):
                        st.markdown(f"[DOI]({paper['Links']['DOI']})")
                    if paper['Links'].get('PMC'):
                        st.markdown(f"[PMC: {paper['Links']['PMC']}](https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['Links']['PMC']})")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit, ChromaDB, and Ollama</div>",
    unsafe_allow_html=True
)