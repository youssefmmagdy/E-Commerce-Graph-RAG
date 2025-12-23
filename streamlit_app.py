
import streamlit as st
import json
from typing import Dict

# Import from your notebook globals (they should be available)
# If not, you may need to save your classes to files

st.set_page_config(
    page_title="E-Commerce Graph-RAG Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛍️ E-Commerce Graph-RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<strong>Welcome!</strong> This AI assistant uses a Knowledge Graph + LLMs to answer questions about products, sellers, and reviews.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### LLM Model")
llm_models = {
    "Mistral 7B": "mistral-7b",
    "Gemma 2B": "gemma-2b",
    "Zephyr 7B": "zephyr-7b"
}
selected_model = st.sidebar.selectbox("Choose LLM", list(llm_models.keys()))

st.sidebar.markdown("### Retrieval Method")
retrieval_modes = {
    "Auto": "auto",
    "Baseline": "baseline",
    "Embeddings": "embeddings",
    "Hybrid": "hybrid"
}
selected_retrieval = st.sidebar.selectbox("Choose Retrieval", list(retrieval_modes.keys()))

show_cypher = st.sidebar.checkbox("Show Cypher Query", value=True)
show_context = st.sidebar.checkbox("Show KG Context", value=True)
show_metrics = st.sidebar.checkbox("Show Metrics", value=True)

# Example queries
st.sidebar.markdown("### 📝 Examples")
examples = [
    "Show me beauty_health products with rating above 4",
    "What are the best products?",
    "Tell me about top sellers"
]

for ex in examples:
    if st.sidebar.button(f"💡 {ex[:30]}..."):
        st.session_state.query = ex

# Main query input
user_query = st.text_input(
    "🔍 Ask a question:",
    value=st.session_state.get('query', ''),
    placeholder="e.g., Show me beauty_health products with rating above 4"
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🚀 Search", type="primary")

if search_button and user_query:
    st.info("⚠️ Note: Full functionality requires running from notebook with initialized connection.")
    st.markdown("This is a demo UI. To use with live data, run the complete pipeline from your notebook.")
    
    # Demo response
    st.markdown('<h2 class="section-header">📊 Query Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Intent", "search")
    with col2:
        st.metric("Retrieval", "baseline")
    with col3:
        st.metric("Results", "10")
    
    st.markdown('<h2 class="section-header">🤖 AI Answer</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="success-box">
    Based on the query results, here are beauty_health products with ratings above 4...
    (This is a demo - run the complete pipeline from notebook for live results)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
<small>Powered by Neo4j + HuggingFace | Milestone 3</small>
</div>
""", unsafe_allow_html=True)
