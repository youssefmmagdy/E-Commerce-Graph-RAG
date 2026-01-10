import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Set environment variable to ensure models are downloaded properly
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")

# Get the base directory (Graph_RAG_M3 folder)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths for persisted artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Model 1: MiniLM (faster, 384 dimensions)
EMBEDDINGS_PATH_M1 = os.path.join(ARTIFACTS_DIR, "embeddings_minilm.npy")
INDEX_PATH_M1 = os.path.join(ARTIFACTS_DIR, "faiss_minilm.index")

# Model 2: MPNET (more accurate, 768 dimensions)
EMBEDDINGS_PATH_M2 = os.path.join(ARTIFACTS_DIR, "embeddings_mpnet.npy")
INDEX_PATH_M2 = os.path.join(ARTIFACTS_DIR, "faiss_mpnet.index")

# Shared chunks metadata
CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.pkl")

# CSV path using absolute path
csv_path = os.path.join(base_dir, 'Ecommerce_KG_Optimized_translated.csv')
df = pd.read_csv(csv_path)

def chunk_text(text, max_length=400, overlap=100):
    chunks = []
    current = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # If adding the new line exceeds max_length → finalize chunk
        if sum(len(x) for x in current) + len(line) > max_length:
            chunk = " ".join(current)
            chunks.append(chunk)

            # --- NEW: create overlap ---
            # Convert chunk into characters and take the last `overlap` chars
            if overlap > 0:
                overlap_text = chunk[-overlap:]
                current = [overlap_text, line]  # start new chunk with overlap + new line
            else:
                current = [line]

        else:
            current.append(line)

    # Add the final chunk
    if current:
        chunks.append(" ".join(current))

    return chunks

all_rows = []
for i, row in df.iterrows():
    # Create a more semantically meaningful text with clear field labels
    # This helps the embedding model understand the structure better
    txt = (
        f"Product Category: {row['product_category_name']}. "
        f"Customer City: {row['customer_city']}. "
        f"Customer State: {row['customer_state']}. "
        f"Order Status: {row['order_status']}. "
        f"Price: {row['price']}. "
        f"Freight Value: {row['freight_value']}. "
        f"Review Score: {row['review_score']}. "
        f"Delivery Delay Days: {row['delivery_delay_days']}. "
        f"Sentiment: {row['sentiment_group']}. "
        f"Review Title: {row['review_comment_title']}. "
        f"Review Message: {row['review_comment_message']}."
    )
    all_rows.append(chunk_text(txt))

# Load both embedding models with proper cache handling
def load_sentence_transformer(model_name: str):
    """Load a SentenceTransformer model with proper error handling for deployment"""
    try:
        # Try loading with cache_folder specified
        cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
        os.makedirs(cache_folder, exist_ok=True)
        model = SentenceTransformer(model_name, cache_folder=cache_folder)
        return model
    except Exception as e:
        print(f"First attempt failed: {e}")
        # Fallback: try without cache folder
        try:
            model = SentenceTransformer(model_name)
            return model
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            raise e2

print("Loading Model 1: MiniLM-L6-v2 (384D, faster)...")
embedder_m1 = load_sentence_transformer("all-MiniLM-L6-v2")

print("Loading Model 2: MPNET-base-v2 (768D, more accurate)...")
embedder_m2 = load_sentence_transformer("all-mpnet-base-v2")

# Flatten all_rows into a single list of chunks with row mapping
all_chunks = []
chunk_to_row = []  # maps chunk index -> original row index

for row_idx, row_chunks in enumerate(all_rows):
    for chunk in row_chunks:
        all_chunks.append(chunk)
        chunk_to_row.append(row_idx)

print(f"Total chunks: {len(all_chunks)} from {len(all_rows)} rows")

# --- PERSISTENCE: Load or Compute Embeddings for BOTH models ---

# Check if chunks metadata exists
if os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        saved_data = pickle.load(f)
        all_chunks = saved_data["chunks"]
        chunk_to_row = saved_data["chunk_to_row"]
    print(f"Loaded chunks metadata: {len(all_chunks)} chunks")
else:
    # Save chunks metadata
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "chunk_to_row": chunk_to_row}, f)
    print(f"Saved chunks metadata")

# --- Model 1: MiniLM embeddings ---
if os.path.exists(EMBEDDINGS_PATH_M1):
    print("Loading MiniLM embeddings from disk...")
    embeddings_m1 = np.load(EMBEDDINGS_PATH_M1)
    print(f"Loaded MiniLM embeddings shape: {embeddings_m1.shape}")
else:
    print("Computing MiniLM embeddings...")
    embeddings_m1 = embedder_m1.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    ).astype("float32")
    np.save(EMBEDDINGS_PATH_M1, embeddings_m1)
    print(f"Saved MiniLM embeddings {embeddings_m1.shape}")

# --- Model 2: MPNET embeddings ---
if os.path.exists(EMBEDDINGS_PATH_M2):
    print("Loading MPNET embeddings from disk...")
    embeddings_m2 = np.load(EMBEDDINGS_PATH_M2)
    print(f"Loaded MPNET embeddings shape: {embeddings_m2.shape}")
else:
    print("Computing MPNET embeddings...")
    embeddings_m2 = embedder_m2.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    ).astype("float32")
    np.save(EMBEDDINGS_PATH_M2, embeddings_m2)
    print(f"Saved MPNET embeddings {embeddings_m2.shape}")

# --- PERSISTENCE: Load or Build FAISS Indexes for BOTH models ---

# Model 1: MiniLM index
if os.path.exists(INDEX_PATH_M1):
    print("Loading MiniLM FAISS index...")
    index_m1 = faiss.read_index(INDEX_PATH_M1)
    print(f"Loaded MiniLM index with {index_m1.ntotal} vectors")
else:
    print("Building MiniLM FAISS index...")
    dim_m1 = embeddings_m1.shape[1]
    index_m1 = faiss.IndexFlatL2(dim_m1)
    index_m1.add(embeddings_m1)
    faiss.write_index(index_m1, INDEX_PATH_M1)
    print(f"Saved MiniLM index with {index_m1.ntotal} vectors")

# Model 2: MPNET index
if os.path.exists(INDEX_PATH_M2):
    print("Loading MPNET FAISS index...")
    index_m2 = faiss.read_index(INDEX_PATH_M2)
    print(f"Loaded MPNET index with {index_m2.ntotal} vectors")
else:
    print("Building MPNET FAISS index...")
    dim_m2 = embeddings_m2.shape[1]
    index_m2 = faiss.IndexFlatL2(dim_m2)
    index_m2.add(embeddings_m2)
    faiss.write_index(index_m2, INDEX_PATH_M2)
    print(f"Saved MPNET index with {index_m2.ntotal} vectors")

# Helper function to build structured results from FAISS search
def _build_results(query, distances, indices, model_name):
    results = []
    seen_rows = set()

    for i, idx in enumerate(indices[0]):
        row_idx = chunk_to_row[idx]
        if row_idx in seen_rows:
            continue
        seen_rows.add(row_idx)

        row = df.iloc[row_idx]
        results.append({
            "source": f"embedding_{model_name}",
            "model": model_name,
            "product_id": str(row.get('product_id', 'N/A')),
            "product_category_name": str(row.get('product_category_name', 'N/A')),
            "customer_city": str(row.get('customer_city', 'N/A')),
            "customer_state": str(row.get('customer_state', 'N/A')),
            "order_status": str(row.get('order_status', 'N/A')),
            "price": float(row.get('price', 0)),
            "freight_value": float(row.get('freight_value', 0)),
            "product_description_lenght": float(row.get('product_description_lenght', 0)),
            "product_photos_qty": float(row.get('product_photos_qty', 0)),
            "delivery_delay_days": int(row.get('delivery_delay_days', 0)),
            "review_score": float(row.get('review_score', 0)),
            "review_comment_message": str(row.get('review_comment_message', 'N/A')),
            "sentiment_group": str(row.get('sentiment_group', 'N/A')),
            "similarity_distance": float(distances[0][i]),
        })
    return results


# Retrieval function for Model 1: MiniLM
def retrieve_minilm(query, k=3):
    """Retrieve using MiniLM-L6-v2 (faster, 384D)"""
    query_emb = embedder_m1.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index_m1.search(query_emb, k)

    print(f"[MiniLM] Top {k} matches: {indices[0]}, Distances: {distances[0]}")
    print("Top k rows:")
    for idx in indices[0]:
        print(df.iloc[chunk_to_row[idx]])
    return _build_results(query, distances, indices, "MiniLM")


# Retrieval function for Model 2: MPNET
def retrieve_mpnet(query, k=3):
    """Retrieve using MPNET-base-v2 (more accurate, 768D)"""
    query_emb = embedder_m2.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index_m2.search(query_emb, k)

    print(f"[MPNET] Top {k} matches: {indices[0]}, Distances: {distances[0]}")
    print("Top k rows:")
    for idx in indices[0]:
        print(df.iloc[chunk_to_row[idx]])
    return _build_results(query, distances, indices, "MPNET")


# Comparison function - returns results from both models
def retrieve_compare(query, k=3):
    """Retrieve using both models for comparison"""
    results_m1 = retrieve_minilm(query, k)
    results_m2 = retrieve_mpnet(query, k)
    return {
        "minilm": results_m1,
        "mpnet": results_m2
    }

# Export functions for use in other modules
def get_embedded_records_minilm(query, k=3):
    """Get records using MiniLM model"""
    return retrieve_minilm(query, k)

def get_embedded_records_mpnet(query, k=3):
    """Get records using MPNET model"""
    return retrieve_mpnet(query, k)

def get_embedded_records(query, k=3):
    """Get records using both models (for comparison)"""
    return retrieve_compare(query, k)

