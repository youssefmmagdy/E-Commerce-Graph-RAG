import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if embeddings artifacts exist, if not run Embeddor.py
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Embedding", "artifacts")
REQUIRED_FILES = [
    "embeddings_minilm.npy",
    
    "faiss_minilm.index",
    
    "chunks.pkl"
]

def check_and_generate_embeddings():
    """Check if embeddings exist, if not generate them by running Embeddor.py"""
    missing_files = []
    
    if not os.path.exists(ARTIFACTS_DIR):
        missing_files = REQUIRED_FILES
    else:
        for f in REQUIRED_FILES:
            if not os.path.exists(os.path.join(ARTIFACTS_DIR, f)):
                missing_files.append(f)
    
    if missing_files:
        st.info(f"🔄 Embeddings not found. Generating embeddings... This may take a few minutes.")
        st.write(f"Missing files: {', '.join(missing_files)}")
        
        embeddor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Embedding", "Embeddor.py")
        
        with st.spinner("Generating embeddings with Embeddor.py..."):
            try:
                result = subprocess.run(
                    [sys.executable, embeddor_path],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                if result.returncode == 0:
                    st.success("✅ Embeddings generated successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Error generating embeddings: {result.stderr}")
                    return False
            except Exception as e:
                st.error(f"❌ Failed to run Embeddor.py: {e}")
                return False
    return True

# Run embedding check at startup
if not check_and_generate_embeddings():
    st.stop()

from Baseline.Baseline import get_baseline_records
from Baseline.EntityExtractor import extract_entities
from Queries.Queries import get_cypher_query_by_number
from LLM.LLM_langchain import (
    rag_answer, rag_compare, display_comparison, 
    create_evaluation_form, MODELS, call_model, build_context, build_prompt
)

# Try to import embedding functions (may fail if notebook not run)
try:
    from Embedding.embeddor import get_embedded_records_minilm
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    EMBEDDINGS_AVAILABLE = False
    st.warning(f"Embeddings not available: {e}")

# =============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# =============================================================================
def build_graph_from_records(records):
    """
    Build a NetworkX graph from retrieved records.
    Creates nodes and edges based on entity relationships.
    """
    G = nx.Graph()
    
    # Node type colors
    node_colors = {
        'Customer': '#4CAF50',      # Green
        'Order': '#2196F3',         # Blue
        'Product': '#FF9800',       # Orange
        'Seller': '#9C27B0',        # Purple
        'Review': '#F44336',        # Red
        'City': '#00BCD4',          # Cyan
        'State': '#795548',         # Brown
        'Category': '#E91E63',      # Pink
    }
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        # Extract entities and create nodes
        customer_id = record.get('customer_id')
        order_id = record.get('order_id')
        product_id = record.get('product_id')
        seller_id = record.get('seller_id')
        review_id = record.get('review_id')
        city = record.get('customer_city')
        state = record.get('customer_state')
        category = record.get('product_category_name')
        
        # Add Customer node
        if customer_id:
            short_id = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            G.add_node(f"C:{short_id}", 
                      label=f"Customer\n{short_id}",
                      node_type='Customer',
                      full_id=customer_id)
        
        # Add Order node
        if order_id:
            short_id = order_id[:8] if len(str(order_id)) > 8 else order_id
            G.add_node(f"O:{short_id}",
                      label=f"Order\n{short_id}",
                      node_type='Order',
                      full_id=order_id,
                      status=record.get('order_status', ''))
        
        # Add Product node
        if product_id:
            short_id = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_node(f"P:{short_id}",
                      label=f"Product\n{short_id}",
                      node_type='Product',
                      full_id=product_id,
                      price=record.get('price', 0))
        
        # Add Seller node
        if seller_id:
            short_id = seller_id[:8] if len(str(seller_id)) > 8 else seller_id
            G.add_node(f"S:{short_id}",
                      label=f"Seller\n{short_id}",
                      node_type='Seller',
                      full_id=seller_id)
        
        # Add Review node
        if review_id:
            short_id = review_id[:8] if len(str(review_id)) > 8 else review_id
            score = record.get('review_score', 'N/A')
            G.add_node(f"R:{short_id}",
                      label=f"Review\n⭐{score}",
                      node_type='Review',
                      full_id=review_id,
                      score=score)
        
        # Add City node
        if city:
            G.add_node(f"City:{city}",
                      label=f"🏙️ {city}",
                      node_type='City')
        
        # Add State node
        if state:
            G.add_node(f"State:{state}",
                      label=f"📍 {state}",
                      node_type='State')
        
        # Add Category node
        if category:
            G.add_node(f"Cat:{category}",
                      label=f"📦 {category[:15]}",
                      node_type='Category')
        
        # Add edges (relationships)
        if customer_id and order_id:
            c_short = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            G.add_edge(f"C:{c_short}", f"O:{o_short}", relationship="PLACED")
        
        if order_id and product_id:
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_edge(f"O:{o_short}", f"P:{p_short}", relationship="CONTAINS")
        
        if product_id and seller_id:
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            s_short = seller_id[:8] if len(str(seller_id)) > 8 else seller_id
            G.add_edge(f"P:{p_short}", f"S:{s_short}", relationship="SOLD_BY")
        
        if order_id and review_id:
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            r_short = review_id[:8] if len(str(review_id)) > 8 else review_id
            G.add_edge(f"R:{r_short}", f"O:{o_short}", relationship="REVIEWS")
        
        if customer_id and city:
            c_short = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            G.add_edge(f"C:{c_short}", f"City:{city}", relationship="LIVES_IN")
        
        if city and state:
            G.add_edge(f"City:{city}", f"State:{state}", relationship="IN_STATE")
        
        if product_id and category:
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_edge(f"P:{p_short}", f"Cat:{category}", relationship="HAS_CATEGORY")
    
    return G


def create_plotly_graph(G):
    """
    Create an interactive Plotly graph visualization from NetworkX graph.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Node type colors
    node_colors = {
        'Customer': '#4CAF50',
        'Order': '#2196F3',
        'Product': '#FF9800',
        'Seller': '#9C27B0',
        'Review': '#F44336',
        'City': '#00BCD4',
        'State': '#795548',
        'Category': '#E91E63',
    }
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get('relationship', ''))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces (one per node type for legend)
    node_traces = []
    
    for node_type in node_colors.keys():
        node_x = []
        node_y = []
        node_text = []
        node_labels = []
        
        for node in G.nodes(data=True):
            if node[1].get('node_type') == node_type:
                x, y = pos[node[0]]
                node_x.append(x)
                node_y.append(y)
                node_labels.append(node[1].get('label', node[0]))
                
                # Build hover text
                hover_info = f"<b>{node_type}</b><br>"
                if 'full_id' in node[1]:
                    hover_info += f"ID: {node[1]['full_id']}<br>"
                if 'status' in node[1] and node[1]['status']:
                    hover_info += f"Status: {node[1]['status']}<br>"
                if 'price' in node[1] and node[1]['price']:
                    hover_info += f"Price: ${node[1]['price']:.2f}<br>"
                if 'score' in node[1]:
                    hover_info += f"Score: {node[1]['score']}<br>"
                node_text.append(hover_info)
        
        if node_x:  # Only create trace if there are nodes of this type
            trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition="bottom center",
                textfont=dict(size=8),
                hovertext=node_text,
                name=node_type,
                marker=dict(
                    color=node_colors[node_type],
                    size=25,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                )
            )
            node_traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces,
                   layout=go.Layout(
                       title=dict(
                           text='🔗 Knowledge Graph Subgraph Visualization',
                           font=dict(size=16)
                       ),
                       showlegend=True,
                       legend=dict(
                           orientation="h",
                           yanchor="bottom",
                           y=1.02,
                           xanchor="center",
                           x=0.5
                       ),
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=60),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',
                       height=500
                   ))
    
    return fig

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="E-Commerce Graph-RAG Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graph.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # Task Selection
    st.subheader("🎯 Task Selection")
    task = st.selectbox(
        "Choose Assistant Type",
        [
            "🔍 Product Search & QA",
            "📦 Ordering Assistant",
            "⭐ Product Recommender",
            "🚚 Delivery Insights",
            "📊 E-Commerce Analytics"
        ]
    )
    
    st.markdown("---")
    
    # Retrieval Method Selection
    st.subheader("🔄 Retrieval Method")
    retrieval_method = st.radio(
        "Select retrieval approach:",
        ["Baseline (Cypher)", "Embeddings (MiniLM)", "Hybrid (All)"],
        index=2  # Default to Hybrid
    )
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 LLM Model")
    model_options = list(MODELS.keys())
    model_names = [MODELS[k]["name"] for k in model_options]
    selected_model = st.selectbox(
        "Choose LLM:",
        model_options,
        format_func=lambda x: MODELS[x]["name"]
    )
    
    compare_models = st.checkbox("Compare all 3 models", value=False)
    
    st.markdown("---")
    
    # Display Options
    st.subheader("👁️ Display Options")
    show_cypher = st.checkbox("Show Cypher Queries", value=True)
    show_entities = st.checkbox("Show Extracted Entities", value=True)
    show_raw_context = st.checkbox("Show Raw KG Context", value=True)
    show_metrics = st.checkbox("Show Response Metrics", value=True)
    show_eval_form = st.checkbox("Show Evaluation Form", value=False)
    show_graph_viz = st.checkbox("Show Graph Visualization", value=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown('<p class="main-header">🛒 E-Commerce Graph-RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Product Search powered by Knowledge Graphs + LLMs</p>', unsafe_allow_html=True)

# =============================================================================
# QUERY INPUT
# =============================================================================
# Initialize session state for query
if "user_query" not in st.session_state:
    st.session_state.user_query = "Electronics in Ibiapine"

# Example queries - matching 10 Cypher queries
examples = [
    "Products by category: Electronics",                    # QUERY_PRODUCTS_BY_CATEGORY
    "Electronics products in Ibiapine",                     # QUERY_PRODUCTS_BY_CATEGORY_AND_CITY
    "Products in São Paulo",                                # QUERY_PRODUCTS_BY_CITY
    "Reviews for product with product_id e481f51cbdc54678b7cc49136f2d6af7",  # QUERY_REVIEWS_FOR_PRODUCT
    "Orders by customer_id 3ce436f183e68e07877b285a838db11a",  # QUERY_ORDERS_BY_CUSTOMER
    "Orders with delivery_delay_days over 5",               # QUERY_ORDERS_WITH_DELAYS
    "Customers in state SP",                                # QUERY_CUSTOMERS_BY_STATE
    "Order with order_id 53cdb2fc8bc7dce0b6741e2150273451",  # QUERY_GET_SPECIFIC_ORDER
    "Customers that bought from seller with seller_id cc419e0650a3c5ba77189a1882b7556a",  # QUERY_CUSTOMERS_BY_SELLER
    "Customers that ordered in São Paulo",                  # QUERY_CUSTOMERS_BY_CITY
]

def set_example_query(query):
    st.session_state.user_query = query

col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "🔍 Enter your query:",
        placeholder="e.g., Electronics in Ibiapine, Products with good reviews, Delayed deliveries...",
        value=st.session_state.user_query,
        key="query_input"
    )
    # Update session state when user types
    st.session_state.user_query = user_query
with col2:
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Example query buttons
with st.expander("💡 Example Queries (10 Options)"):
    example_cols = st.columns(2)
    for i, example in enumerate(examples):
        with example_cols[i % 2]:
            st.button(example, key=f"ex_{i}", on_click=set_example_query, args=(example,))

# =============================================================================
# PROCESSING
# =============================================================================
if search_button or user_query:
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 LLM Response", 
        "📊 KG Context", 
        "🔗 Graph Visualization",
        "⚡ Cypher & Entities",
        "📈 Model Comparison"
    ])
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    with st.spinner("🔍 Retrieving data from Knowledge Graph..."):
        all_records = []
        retrieval_info = {}
        
        # Baseline retrieval
        if retrieval_method in ["Baseline (Cypher)", "Hybrid (All)"]:
            try:
                # Attempt to extract query_number from user_query if it starts with a number and dot (e.g., "2. ...")
                import re
                match = re.match(r"(\d+)\.", user_query.strip())
                if match:
                    query_number = int(match.group(1))
                    baseline_records = get_baseline_records(user_query, query_number)
                else:
                    # Auto-detect query type based on entities
                    baseline_records = get_baseline_records(user_query)

                all_records.extend(baseline_records)
                retrieval_info["baseline"] = len(baseline_records)
            except Exception as e:
                st.error(f"Baseline retrieval error: {e}")
                import traceback
                st.code(traceback.format_exc())
                retrieval_info["baseline"] = 0
        
        # Embedding retrieval
        if EMBEDDINGS_AVAILABLE:
            if retrieval_method in ["Embeddings (MiniLM)", "Hybrid (All)"]:
                try:
                    minilm_records = get_embedded_records_minilm(user_query, k=3)
                    all_records.extend(minilm_records)
                    retrieval_info["minilm"] = len(minilm_records)
                except Exception as e:
                    st.warning(f"MiniLM retrieval error: {e}")
                    retrieval_info["minilm"] = 0
            
            # if retrieval_method in ["Embeddings (MPNET)", "Hybrid (All)"]:
            #     try:
            #         mpnet_records = get_embedded_records_mpnet(user_query, k=3)
            #         all_records.extend(mpnet_records)
            #         retrieval_info["mpnet"] = len(mpnet_records)
            #     except Exception as e:
            #         st.warning(f"MPNET retrieval error: {e}")
            #         retrieval_info["mpnet"] = 0
    
    # =========================================================================
    # TAB 1: LLM RESPONSE
    # =========================================================================
    with tab1:
        st.subheader("🤖 LLM Response")
        
        if len(all_records) == 0:
            st.warning("No records found. Try a different query.")
        else:
            # Retrieval summary
            st.info(f"📊 Retrieved **{len(all_records)}** total records: " + 
                   ", ".join([f"{k}: {v}" for k, v in retrieval_info.items()]))
            
            if compare_models:
                # Compare all models
                st.markdown("### Comparing 3 LLM Models")
                
                comparison_cols = st.columns(3)
                
                context = build_context(all_records)
                prompt = build_prompt(context, user_query)
                
                for i, model_key in enumerate(MODELS.keys()):
                    with comparison_cols[i]:
                        st.markdown(f"**{MODELS[model_key]['name']}**")
                        with st.spinner(f"Calling {MODELS[model_key]['name']}..."):
                            result = call_model(model_key, prompt)
                        
                        if result.success:
                            st.success(result.response)
                            if show_metrics:
                                st.caption(f"⏱️ {result.response_time_ms:.0f}ms | 📝 ~{result.output_tokens} tokens")
                        else:
                            st.error(f"Error: {result.error}")
            else:
                # Single model response
                st.markdown(f"### Response from {MODELS[selected_model]['name']}")
                
                with st.spinner(f"Generating response with {MODELS[selected_model]['name']}..."):
                    start_time = time.time()
                    response = rag_answer(all_records, user_query, model_key=selected_model)
                    response_time = (time.time() - start_time) * 1000
                
                st.markdown(response)
                
                if show_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Response Time", f"{response_time:.0f}ms")
                    with col2:
                        st.metric("📄 Records Used", len(all_records))
                    with col3:
                        st.metric("🔤 Response Length", f"{len(response)} chars")
    
    # =========================================================================
    # TAB 2: KG CONTEXT
    # =========================================================================
    with tab2:
        st.subheader("📊 Retrieved Knowledge Graph Data")
        
        if show_raw_context and len(all_records) > 0:
            for i, record in enumerate(all_records, 1):
                with st.expander(f"Record {i}", expanded=(i <= 3)):
                    if isinstance(record, dict):
                        # Convert all values to strings to avoid Arrow serialization issues
                        record_str = {k: str(v) if v is not None else "" for k, v in record.items()}
                        df = pd.DataFrame([record_str]).T
                        df.columns = ["Value"]
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.json(dict(record) if hasattr(record, 'keys') else str(record))
        
        # Summary statistics
        if len(all_records) > 0 and isinstance(all_records[0], dict):
            st.markdown("### 📈 Summary Statistics")
            
            try:
                df_records = pd.DataFrame(all_records)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'price' in df_records.columns:
                        st.metric("💰 Avg Price", f"${df_records['price'].mean():.2f}")
                    if 'review_score' in df_records.columns:
                        st.metric("⭐ Avg Review", f"{df_records['review_score'].mean():.1f}/5")
                
                with col2:
                    if 'category' in df_records.columns:
                        st.metric("📦 Categories", df_records['category'].nunique())
                    if 'city' in df_records.columns:
                        st.metric("🏙️ Cities", df_records['city'].nunique())
            except:
                pass
    
    # =========================================================================
    # TAB 3: GRAPH VISUALIZATION
    # =========================================================================
    with tab3:
        st.subheader("🔗 Knowledge Graph Visualization")
        
        if show_graph_viz and len(all_records) > 0:
            with st.spinner("Building graph visualization..."):
                # Build NetworkX graph from records
                G = build_graph_from_records(all_records)
                
                if len(G.nodes()) > 0:
                    # Graph statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔵 Nodes", len(G.nodes()))
                    with col2:
                        st.metric("🔗 Edges", len(G.edges()))
                    with col3:
                        # Count node types
                        node_types = set(nx.get_node_attributes(G, 'node_type').values())
                        st.metric("📊 Node Types", len(node_types))
                    with col4:
                        # Count connected components
                        if len(G.nodes()) > 0:
                            components = nx.number_connected_components(G)
                            st.metric("🧩 Components", components)
                    
                    # Create and display Plotly graph
                    fig = create_plotly_graph(G)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Node type breakdown
                    with st.expander("📊 Node Type Breakdown"):
                        node_type_counts = {}
                        for node, data in G.nodes(data=True):
                            ntype = data.get('node_type', 'Unknown')
                            node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
                        
                        type_df = pd.DataFrame([
                            {"Node Type": k, "Count": v, "Icon": {
                                'Customer': '👤',
                                'Order': '📦',
                                'Product': '🛍️',
                                'Seller': '🏪',
                                'Review': '⭐',
                                'City': '🏙️',
                                'State': '📍',
                                'Category': '🏷️'
                            }.get(k, '•')}
                            for k, v in node_type_counts.items()
                        ])
                        st.dataframe(type_df, use_container_width=True, hide_index=True)
                    
                    # Edge/Relationship breakdown
                    with st.expander("🔗 Relationship Breakdown"):
                        edge_type_counts = {}
                        for u, v, data in G.edges(data=True):
                            rel = data.get('relationship', 'RELATED')
                            edge_type_counts[rel] = edge_type_counts.get(rel, 0) + 1
                        
                        edge_df = pd.DataFrame([
                            {"Relationship": k, "Count": v}
                            for k, v in edge_type_counts.items()
                        ])
                        st.dataframe(edge_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No graph nodes could be extracted from the records.")
        else:
            if not show_graph_viz:
                st.info("Graph visualization is disabled. Enable it in the sidebar.")
            else:
                st.warning("No records to visualize. Try a different query.")
    
    # =========================================================================
    # TAB 4: CYPHER & ENTITIES
    # =========================================================================
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if show_entities:
                st.subheader("🏷️ Extracted Entities")
                try:
                    entities = extract_entities(user_query)
                    
                    # entities is a list of (text, label) tuples
                    entity_df = pd.DataFrame([
                        {"Type": label, "Value": text}
                        for text, label in entities
                    ])
                    
                    if not entity_df.empty:
                        st.dataframe(entity_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No entities extracted from query")
                except Exception as e:
                    st.error(f"Entity extraction error: {e}")
        
        with col2:
            if show_cypher:
                st.subheader("⚡ Cypher Query Executed")
                
                cypher_query = get_cypher_query_by_number(2)
                st.code(cypher_query, language="cypher")
                
                st.caption("Query 2: Products by category and city (baseline retrieval)")
    
    # =========================================================================
    # TAB 5: MODEL COMPARISON
    # =========================================================================
    with tab5:
        st.subheader("📈 Model Comparison Analysis")
        
        if len(all_records) > 0:
            if st.button("🔄 Run Full Comparison", type="primary"):
                
                with st.spinner("Running comparison across all 3 models..."):
                    # Use rag_compare to compare all models
                    comparison_result = rag_compare(all_records, user_query)
                
                # Display comparison table using ComparisonResult
                st.markdown("### 📊 Quantitative Metrics")
                
                metrics_data = []
                for resp in comparison_result.responses:
                    metrics_data.append({
                        "Model": resp.model_name,
                        "Response Time (ms)": f"{resp.response_time_ms:.0f}",
                        "Input Tokens (est)": resp.input_tokens,
                        "Output Tokens (est)": resp.output_tokens,
                        "Status": "✓ Success" if resp.success else f"✗ {resp.error}"
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                
                # Visual comparison chart
                st.markdown("### ⏱️ Response Time Comparison")
                chart_data = pd.DataFrame({
                    "Model": [r.model_name for r in comparison_result.responses],
                    "Response Time (ms)": [r.response_time_ms for r in comparison_result.responses]
                })
                st.bar_chart(chart_data.set_index("Model"))
                
                # Token usage comparison
                st.markdown("### 📝 Token Usage Comparison")
                token_data = pd.DataFrame({
                    "Model": [r.model_name for r in comparison_result.responses],
                    "Input Tokens": [r.input_tokens for r in comparison_result.responses],
                    "Output Tokens": [r.output_tokens for r in comparison_result.responses]
                })
                st.bar_chart(token_data.set_index("Model"))
                
                # Model Responses side by side
                st.markdown("### 💬 Model Responses")
                response_cols = st.columns(len(comparison_result.responses))
                
                for i, resp in enumerate(comparison_result.responses):
                    with response_cols[i]:
                        st.markdown(f"**{resp.model_name}**")
                        if resp.success:
                            st.success(resp.response)
                        else:
                            st.error(f"Error: {resp.error}")
                
                # Store comparison result in session state for evaluation
                st.session_state['last_comparison'] = comparison_result
                
                # Display comparison summary (console-style)
                with st.expander("📋 Raw Comparison Output (display_comparison)"):
                    # Create a text representation similar to display_comparison
                    output = []
                    output.append("=" * 80)
                    output.append("LLM COMPARISON RESULTS (LangChain)")
                    output.append("=" * 80)
                    output.append(f"Query: {comparison_result.query}")
                    output.append(f"Context: {comparison_result.context_summary}")
                    output.append("=" * 80)
                    
                    for resp in comparison_result.responses:
                        output.append(f"\n{'─' * 40}")
                        output.append(f"MODEL: {resp.model_name}")
                        output.append(f"{'─' * 40}")
                        
                        if resp.success:
                            output.append(f"Response Time: {resp.response_time_ms:.0f}ms")
                            output.append(f"Tokens (est): {resp.input_tokens} input, {resp.output_tokens} output")
                            output.append(f"\nResponse:\n{resp.response}")
                        else:
                            output.append(f"ERROR: {resp.error}")
                    
                    output.append("\n" + "=" * 80)
                    output.append("QUANTITATIVE SUMMARY")
                    output.append("=" * 80)
                    output.append(f"{'Model':<20} {'Time (ms)':<12} {'In Tokens':<12} {'Out Tokens':<12} {'Status':<10}")
                    output.append("-" * 66)
                    
                    for resp in comparison_result.responses:
                        status = "✓ OK" if resp.success else "✗ FAIL"
                        output.append(f"{resp.model_name:<20} {resp.response_time_ms:<12.0f} {resp.input_tokens:<12} {resp.output_tokens:<12} {status:<10}")
                    
                    st.code("\n".join(output), language="text")
            
            # Evaluation Form Section
            st.markdown("---")
            st.markdown("### 📝 Qualitative Evaluation Form")
            
            if 'last_comparison' in st.session_state and show_eval_form:
                comparison_result = st.session_state['last_comparison']
                
                # Generate and display the evaluation form
                with st.expander("📄 Printable Evaluation Form (create_evaluation_form)", expanded=False):
                    eval_form = create_evaluation_form(comparison_result)
                    st.code(eval_form, language="text")
                    
                    # Download button for the form
                    st.download_button(
                        label="📥 Download Evaluation Form",
                        data=eval_form,
                        file_name=f"evaluation_form_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                
                # Interactive evaluation form
                st.markdown("#### 🎯 Interactive Evaluation")
                st.markdown(f"**Query:** {comparison_result.query}")
                st.markdown(f"**Context:** {comparison_result.context_summary}")
                
                criteria = ["Relevance", "Accuracy", "Completeness", "Conciseness", "Naturalness", "Groundedness"]
                criteria_descriptions = {
                    "Relevance": "Does it answer the question?",
                    "Accuracy": "Is it correct based on KG data?",
                    "Completeness": "Does it cover all relevant info?",
                    "Conciseness": "Is it free of unnecessary info?",
                    "Naturalness": "Is it fluent and readable?",
                    "Groundedness": "Does it avoid hallucinations?"
                }
                
                # Initialize scores in session state
                if 'eval_scores' not in st.session_state:
                    st.session_state['eval_scores'] = {}
                
                for resp in comparison_result.responses:
                    with st.expander(f"🤖 Evaluate: {resp.model_name}", expanded=True):
                        # Show truncated response
                        response_preview = resp.response[:500] + "..." if len(resp.response) > 500 else resp.response
                        st.markdown(f"**Response Preview:**\n> {response_preview}")
                        
                        st.markdown("**Rate each criterion (1-5):**")
                        score_cols = st.columns(6)
                        
                        model_scores = {}
                        for j, criterion in enumerate(criteria):
                            with score_cols[j]:
                                score = st.slider(
                                    f"{criterion}",
                                    min_value=1,
                                    max_value=5,
                                    value=3,
                                    key=f"eval_{resp.model_key}_{criterion}",
                                    help=criteria_descriptions[criterion]
                                )
                                model_scores[criterion] = score
                        
                        total_score = sum(model_scores.values())
                        st.metric(f"Total Score for {resp.model_name}", f"{total_score}/30")
                        st.session_state['eval_scores'][resp.model_key] = model_scores
                
                # Summary of all evaluations
                if st.session_state.get('eval_scores'):
                    st.markdown("#### 🏆 Evaluation Summary")
                    
                    summary_data = []
                    for resp in comparison_result.responses:
                        if resp.model_key in st.session_state['eval_scores']:
                            scores = st.session_state['eval_scores'][resp.model_key]
                            total = sum(scores.values())
                            summary_data.append({
                                "Model": resp.model_name,
                                **scores,
                                "Total": total
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        
                        # Find best model
                        best_idx = summary_df["Total"].idxmax()
                        best_model = summary_df.loc[best_idx, "Model"]
                        best_score = summary_df.loc[best_idx, "Total"]
                        
                        st.success(f"🏆 **Best Model:** {best_model} with score {best_score}/30")
                        
                        # Notes section
                        st.text_area(
                            "📝 Evaluator Notes",
                            placeholder="Add your notes about the comparison here...",
                            key="evaluator_notes"
                        )
                        
                        evaluator_name = st.text_input("Evaluator Name", key="evaluator_name")
                        
                        if st.button("💾 Save Evaluation"):
                            eval_result = {
                                "query": comparison_result.query,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "evaluator": evaluator_name,
                                "notes": st.session_state.get("evaluator_notes", ""),
                                "scores": st.session_state['eval_scores'],
                                "best_model": best_model
                            }
                            st.json(eval_result)
                            st.success("Evaluation saved!")
            
            elif not show_eval_form:
                st.info("💡 Enable 'Show Evaluation Form' in the sidebar to see the evaluation form.")
            else:
                st.info("👆 Click 'Run Full Comparison' above to generate comparison results for evaluation.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>🎓 ACL Milestone 3 - Graph-RAG E-Commerce Assistant</p>
    <p>Built with Streamlit | Neo4j | HuggingFace LLMs</p>
</div>
""", unsafe_allow_html=True)
