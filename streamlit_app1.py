from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import numpy as np
from typing import Dict, Tuple
from typing import Dict
import json
import streamlit as st
import spacy
import re
from typing import Dict, List, Optional
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
import time
token = "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"


class LLMInterface:
    """LLM interface with fallback for different API formats"""

    AVAILABLE_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma-2b": "google/gemma-2-2b-it",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",

    }

    # Models that support chat API
    CHAT_MODELS = ["mistralai/Mistral-7B-Instruct-v0.2",
                   "google/gemma-2-2b-it", "HuggingFaceH4/zephyr-7b-beta"]

    def __init__(self, model_name="mistral-7b", token=None):
        self.model_path = self.AVAILABLE_MODELS[model_name]
        self.model_name = model_name
        self.client = InferenceClient(token=token)

        # Determine which API to use
        self.use_chat_api = self.model_path in self.CHAT_MODELS

        print(
            f"✓ LLM ready: {self.model_path} (API: {'chat' if self.use_chat_api else 'text_generation'})")

    def generate_answer(self, context: str, query: str, max_tokens=512) -> Dict:
        """Generate answer using appropriate API format"""
        prompt = self._build_structured_prompt(context, query)
        start = time.time()

        try:
            if self.use_chat_api:
                # Use chat completions API (Mistral, Gemma)
                answer = self._generate_chat(prompt, max_tokens)
            else:
                # Use text generation API (Phi-3.5 and others)
                answer = self._generate_text(prompt, max_tokens)

            return {
                "answer": answer,
                "model": self.model_name,
                "generation_time": time.time() - start,
                "success": True
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "model": self.model_name,
                "generation_time": time.time() - start,
                "success": False,
                "error": str(e)
            }

    def _generate_chat(self, prompt: str, max_tokens: int) -> str:
        """Generate using chat completions API"""
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "system", "content": "You are a helpful E-commerce assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        """Generate using text generation API"""
        response = self.client.text_generation(
            prompt,
            model=self.model_path,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        return response.strip() if isinstance(response, str) else str(response).strip()

    def _build_structured_prompt(self, context: str, query: str) -> str:
        """Requirement 3.b: Structured prompt"""
        persona = """You are an intelligent E-Commerce marketplace assistant.
You analyze customer reviews, product performance, and delivery reliability.
You provide accurate, data-driven answers based on the knowledge graph."""

        context_section = f"""CONTEXT FROM KNOWLEDGE GRAPH:
{context}"""

        task = """TASK:
Answer the user's question using ONLY the information provided in the context above.
- Be specific and cite data from the context when applicable
- If the context doesn't contain relevant information, say "I don't have enough information to answer that"
- Keep your answer concise but informative
- Use numbers and statistics from the context when available"""

        question_section = f"""USER QUESTION:
{query}"""

        full_prompt = f"""{persona}

{context_section}

{task}

{question_section}

ANSWER:"""

        return full_prompt


class GraphRAGPipeline:
    """Complete end-to-end Graph-RAG pipeline"""

    def __init__(self, neo4j_conn, embedding_model='all-MiniLM-L6-v2',
                 llm_model='mistral-7b', hf_token: Optional[str] = None):

        self.conn = neo4j_conn
        self.retriever = GraphRAGRetriever(neo4j_conn, embedding_model)
        self.llm = LLMInterface(llm_model, token=hf_token)
        self.db_categories = get_db_categories(neo4j_conn)

        print(f"✓ Pipeline initialized with {llm_model}")

    def process_query(self, user_query: str, use_embeddings=True,
                      retrieval_mode='auto', verbose=True) -> Dict:

        if verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING QUERY: {user_query}")
            print(f"{'='*70}")

        # STEP 1: Preprocessing
        if verbose:
            print("\n[1] Intent Classification...")
        intent = classify_intent(user_query)
        if verbose:
            print(f"    Intent: {intent}")

        if verbose:
            print("\n[2] Entity Extraction...")
        entities = extract_entities(user_query, intent, self.db_categories)
        if verbose:
            print(f"    Entities: {entities}")

        # STEP 2: Graph Retrieval
        if verbose:
            print("\n[3] Graph Retrieval...")

        if retrieval_mode == 'baseline':
            retrieval_results = self.retriever.retrieve_baseline(
                intent, entities)
        elif retrieval_mode == 'embeddings':
            retrieval_results = self.retriever.retrieve_embeddings(user_query)
        elif retrieval_mode == 'hybrid':
            retrieval_results = self.retriever.retrieve_hybrid(
                user_query, intent, entities)
        else:  # 'auto' mode
            retrieval_results = self.retriever.retrieve(
                user_query, intent, entities, use_embeddings=use_embeddings
            )

        if verbose:
            print(f"    Method: {retrieval_results['method']}")
            print(f"    Results: {retrieval_results['count']} items")
            if 'fallback_reason' in retrieval_results:
                print(f"    Fallback: {retrieval_results['fallback_reason']}")

        # STEP 3: Format context for LLM (Requirement 3.a)
        if verbose:
            print("\n[4] Formatting context for LLM...")
        context = self._format_context(retrieval_results)

        # STEP 4: Generate LLM answer
        if verbose:
            print("\n[5] Generating LLM answer...")
        llm_response = self.llm.generate_answer(context, user_query)

        if verbose:
            print(
                f"    Generation time: {llm_response['generation_time']:.2f}s")
            print(f"    Success: {llm_response['success']}")

        return {
            'query': user_query,
            'intent': intent,
            'entities': entities,
            'retrieval': retrieval_results,
            'context': context,
            'llm_response': llm_response,
            'final_answer': llm_response['answer']
        }

    def _format_context(self, retrieval_results: Dict) -> str:
        """Requirement 3.a: Combine KG results"""
        method = retrieval_results.get('method')

        # Handle hybrid retrieval
        if method == 'hybrid':
            baseline_results = retrieval_results.get(
                'baseline', {}).get('results', [])
            embedding_results = retrieval_results.get(
                'embeddings', {}).get('results', [])

            context_parts = []

            if baseline_results:
                context_parts.append("=== STRUCTURED QUERY RESULTS ===")
                for i, result in enumerate(baseline_results[:10], 1):
                    context_parts.append(
                        f"{i}. {self._format_result_item(result)}")

            if embedding_results:
                context_parts.append("\n=== SEMANTIC SEARCH RESULTS ===")
                for i, result in enumerate(embedding_results[:10], 1):
                    score = result.get('score', 0)
                    context_parts.append(
                        f"{i}. [Similarity: {score:.3f}] {self._format_result_item(result)}"
                    )

            return "\n".join(context_parts) if context_parts else "No relevant information found."

        # Handle single method retrieval
        results = retrieval_results.get('results', [])

        if not results:
            return "No relevant information found in the knowledge graph."

        context_parts = []

        if method == 'embeddings':
            context_parts.append("=== SEMANTIC SEARCH RESULTS ===")
            for i, result in enumerate(results[:10], 1):
                score = result.get('score', 0)
                context_parts.append(
                    f"{i}. [Similarity: {score:.3f}] {self._format_result_item(result)}"
                )
        else:  # baseline
            context_parts.append("=== QUERY RESULTS ===")
            for i, result in enumerate(results[:10], 1):
                context_parts.append(
                    f"{i}. {self._format_result_item(result)}")

        return "\n".join(context_parts)

    def _format_result_item(self, result: Dict) -> str:
        """Format single result item for context"""
        parts = []

        if 'product_id' in result:
            parts.append(f"Product: {result['product_id']}")
        if 'category' in result:
            parts.append(f"Category: {result['category']}")
        if 'avg_rating' in result:
            parts.append(f"Rating: {result['avg_rating']:.2f}")
        if 'review_count' in result:
            parts.append(f"Reviews: {result['review_count']}")
        if 'order_count' in result:
            parts.append(f"Orders: {result['order_count']}")
        if 'total_sales' in result:
            parts.append(f"Sales: {result['total_sales']}")
        if 'title' in result:
            parts.append(f"Review Title: {result['title']}")
        if 'message' in result and result['message']:
            message = result['message'][:100]
            parts.append(f"Review: {message}...")
        if 'score' in result and 'title' not in result:
            parts.append(f"Score: {result['score']}")

        return " | ".join(parts) if parts else str(result)


def compare_llm_models(neo4j_conn, test_queries: List[str],
                       hf_token: Optional[str] = None) -> Dict:
    """
    Compare different LLM models (Requirement 3.c)
    You must test at least 3 different LLMs

    Args:
        neo4j_conn: Neo4j connection
        test_queries: List of test queries
        hf_token: HuggingFace API token

    Returns:
        Comparison results with quantitative and qualitative metrics
    """

    # Select 3+ models to compare (all FREE)
    llm_models = [
        'mistral-7b',      # Strong general performance
        'gemma-2b',        # Fast, good for quick responses
        'phi-3.5',         # Microsoft model, good reasoning
        # 'llama-3.2-3b',  # Meta model (optional 4th)
    ]

    print("\n" + "="*70)
    print("LLM MODEL COMPARISON")
    print(f"Testing {len(llm_models)} models on {len(test_queries)} queries")
    print("="*70)

    results = {}

    for model_name in llm_models:
        print(f"\n{'='*70}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*70}")

        try:
            # Initialize pipeline with this LLM
            pipeline = GraphRAGPipeline(
                neo4j_conn,
                embedding_model='all-MiniLM-L6-v2',
                llm_model=model_name,
                hf_token=hf_token
            )

            model_results = []

            for i, query in enumerate(test_queries, 1):
                print(f"\n[Query {i}/{len(test_queries)}] {query}")

                # Process query
                response = pipeline.process_query(
                    query, retrieval_mode='auto', verbose=False)

                # Extract metrics
                model_results.append({
                    'query': query,
                    'answer': response['final_answer'],
                    'retrieval_method': response['retrieval']['method'],
                    'retrieval_count': response['retrieval']['count'],
                    'generation_time': response['llm_response']['generation_time'],
                    'success': response['llm_response']['success']
                })

                # Show answer preview
                answer_preview = response['final_answer'][:150]
                print(f"  Answer: {answer_preview}...")
                print(
                    f"  Time: {response['llm_response']['generation_time']:.2f}s")

            results[model_name] = model_results
            print(f"\n✓ Completed testing {model_name}")

        except Exception as e:
            print(f"\n⚠️ Error testing {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Calculate statistics (Requirement 3.d: Quantitative metrics)
    print("\n" + "="*70)
    print("QUANTITATIVE COMPARISON")
    print("="*70)

    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "error" in model_results:
            print(f"\n{model_name}: ERROR - {model_results['error']}")
            continue

        if not model_results:
            continue

        # Calculate metrics
        avg_time = sum(r['generation_time']
                       for r in model_results) / len(model_results)
        success_rate = sum(
            1 for r in model_results if r['success']) / len(model_results) * 100
        avg_answer_length = sum(len(r['answer'])
                                for r in model_results) / len(model_results)

        print(f"\n{model_name}:")
        print(f"  Avg Generation Time: {avg_time:.2f}s")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Answer Length: {avg_answer_length:.0f} chars")

    return results


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, params=None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]


class GraphRAGRetriever:
    """Main retrieval system combining baseline and embeddings"""

    def __init__(self, neo4j_conn, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize Graph-RAG retriever
        Args:
            neo4j_conn: Neo4jConnection instance
            embedding_model: Name of embedding model to use
        """
        self.conn = neo4j_conn
        self.cypher_lib = CypherQueryLibrary()
        self.embedder = InputEmbedder(embedding_model)
        self.embedding_retrieval = EmbeddingRetrieval(
            neo4j_conn, self.embedder)

    def retrieve_baseline(self, intent: str, entities: Dict) -> Dict:
        """
        Baseline retrieval using Cypher queries
        Args:
            intent: Classified intent
            entities: Extracted entities
        Returns:
            dict with query, params, and results
        """
        query, params = self.cypher_lib.get_query(intent, entities)
        results = self.conn.query(query, params)

        return {
            'method': 'baseline',
            'query': query,
            'params': params,
            'results': results,
            'count': len(results) if results else 0
        }

    def retrieve_embeddings(self, query_text: str, top_k=10) -> Dict:
        """
        Embedding-based retrieval using semantic search
        Args:
            query_text: Original user query
            top_k: Number of results
        Returns:
            dict with results
        """
        results = self.embedding_retrieval.semantic_search(query_text, top_k)

        return {
            'method': 'embeddings',
            'query_text': query_text,
            'results': results,
            'count': len(results) if results else 0
        }

    # ========================================================================
    # YOUR retrieve() FUNCTION - THIS IS THE MAIN ENTRY POINT
    # ========================================================================

    def retrieve(self, query_text: str, intent: Optional[str],
                 entities: Dict, use_embeddings=True) -> Dict:
        """
        Main retrieval function with intelligent fallback logic

        Args:
            query_text: Original user query
            intent: Classified intent (None if classification failed)
            entities: Extracted entities
            use_embeddings: Whether to use embedding fallback

        Returns:
            dict with retrieval results and metadata
        """
        # Case 1: Intent failed → embeddings
        if intent is None:
            print("⚠️ Intent classification failed → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'intent_failure'
            return result

        # Case 2: Category missing → embeddings fallback
        if entities.get("category") is None and use_embeddings:
            print("⚠️ No category extracted → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'missing_category'
            return result

        # Case 3: Try baseline first
        print(f"✓ Using baseline retrieval for intent: {intent}")
        baseline = self.retrieve_baseline(intent, entities)

        # If baseline returns empty → embeddings fallback
        if baseline["count"] == 0 and use_embeddings:
            print("⚠️ Baseline returned no results → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'empty_results'
            result['attempted_baseline'] = baseline
            return result

        # Baseline succeeded
        return baseline

    def retrieve_hybrid(self, query_text: str, intent: Optional[str],
                        entities: Dict, top_k=10) -> Dict:
        """
        Combined retrieval: baseline + embeddings
        Useful for comprehensive results or comparison

        Args:
            query_text: Original user query
            intent: Classified intent
            entities: Extracted entities
            top_k: Number of embedding results

        Returns:
            Combined results from both methods
        """
        baseline_results = self.retrieve_baseline(
            intent, entities) if intent else None
        embedding_results = self.retrieve_embeddings(query_text, top_k)

        # Combine results
        combined = {
            'method': 'hybrid',
            'baseline': baseline_results,
            'embeddings': embedding_results,
            'baseline_count': baseline_results['count'] if baseline_results else 0,
            'embeddings_count': embedding_results['count'],
            'total_count': (baseline_results['count'] if baseline_results else 0) + embedding_results['count']
        }

        return combined


# ============================================================================
# SETUP & INITIALIZATION HELPER
# ============================================================================

class GraphRAGSetup:
    """Helper class for one-time setup of embeddings"""

    def __init__(self, retriever: GraphRAGRetriever):
        self.retriever = retriever

    def initialize_embeddings(self, skip_if_exists=True):
        """
        Run one-time setup for embeddings
        Args:
            skip_if_exists: If True, skip if embeddings already exist
        """
        print("=" * 60)
        print("INITIALIZING GRAPH-RAG EMBEDDINGS")
        print("=" * 60)

        # Check if embeddings exist
        if skip_if_exists:
            check_query = """
            MATCH (p:Product)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) as count
            LIMIT 1
            """
            result = self.retriever.conn.query(check_query)
            if result and result[0]['count'] > 0:
                print("✓ Embeddings already exist. Skipping initialization.")
                return

        # Step 1: Create node embeddings
        print("\n[1/3] Creating node embeddings...")
        self.retriever.embedding_retrieval.create_node_embeddings(
            'Product',
            'product_category_name'
        )

        # Step 2: Create feature vector embeddings (optional)
        print("\n[2/3] Creating feature vector embeddings...")
        self.retriever.embedding_retrieval.create_feature_vector_embeddings(
            include_category=False)  # Avoid data leakage
        # )

        # Step 3: Create vector index
        print("\n[3/3] Creating vector index...")
        self.retriever.embedding_retrieval.create_vector_index()

        print("\n" + "=" * 60)
        print("✓ INITIALIZATION COMPLETE")
        print("=" * 60)


class InputEmbedder:
    """Handles input text embedding for semantic similarity search"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def embed_text(self, text):
        """Embed single text string"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def embed_batch(self, texts):
        """Embed multiple texts efficiently"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings


def classify_intent(text: str) -> str:
    """
    Classify user intent from query text
    Returns: intent string or None if unknown
    """
    text = text.lower()

    intent_rules = {
        "recommendation": ["recommend", "suggest", "should i buy", "best for me"],
        "reviews": ["review", "reviews", "opinion", "feedback", "sentiment", "comment"],
        "delivery_analysis": ["delay", "delivery", "shipping", "late", "on time"],
        "seller_analysis": ["seller", "vendor", "merchant", "store", "sold by"],
        "ranking": ["best", "top", "highest", "lowest", "worst", "rank"],
        "analytics": ["average", "trend", "statistics", "distribution", "impact", "analysis", "stats"],
        "qa": ["how many", "what is", "which", "count", "total"]
    }

    for intent, keywords in intent_rules.items():
        if any(k in text for k in keywords):
            return intent

    # Default to search if keywords present, None otherwise
    if any(word in text for word in ["find", "show", "get", "search", "products", "items"]):
        return "search"

    return None  # Unknown intent - will trigger embedding fallback


nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str, intent: str, db_categories: Optional[List[str]] = None) -> Dict:
    """
    Extract entities from user query
    Args:
        text: User query text
        intent: Classified intent (affects which entities to prioritize)
        db_categories: List of valid categories from database
    Returns:
        Dictionary of extracted entities
    """
    text_l = text.lower()
    doc = nlp(text_l)

    entities = {
        "category": None,
        "seller": None,
        "city": None,
        "state": None,
        "rating_min": None,
        "price_max": None,
        "delivery_delay": None,
        "quantity": None,
        "date": None,
        "sentiment": None
    }

    # ---------- Location Entities ----------
    for ent in doc.ents:
        if ent.label_ == "GPE":
            # Try to determine if it's a city or state
            if not entities["city"]:
                # Capitalize for DB matching
                entities["city"] = ent.text.title()
        elif ent.label_ == "ORG":
            entities["seller"] = ent.text
        elif ent.label_ in ["DATE", "TIME"]:
            entities["date"] = ent.text

    # ---------- Category Matching (DB-driven) ----------
    if db_categories:
        # Exact match first
        for cat in db_categories:
            if cat.lower() in text_l:
                entities["category"] = cat
                break

        # If no exact match, try partial/fuzzy matching
        if not entities["category"]:
            for cat in db_categories:
                # Check if query contains part of category name
                cat_words = cat.lower().split('_')
                if any(word in text_l for word in cat_words):
                    entities["category"] = cat
                    break

    # ---------- Rating Extraction ----------
    rating_patterns = [
        r"(rating|rated|score)\s*(above|over|at least|>=?|minimum)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(stars?|rating)",
        r"(above|over)\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in rating_patterns:
        match = re.search(pattern, text_l)
        if match:
            # Extract the numeric value (it may be in different groups)
            numbers = [g for g in match.groups(
            ) if g and g.replace('.', '').isdigit()]
            if numbers:
                entities["rating_min"] = float(numbers[0])
                break

    # ---------- Price Extraction ----------
    price_patterns = [
        r"(under|below|less than|max|maximum)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$\s*(\d+(?:\.\d+)?)",
        r"price\s*(?:of|at)?\s*\$?\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text_l)
        if match:
            numbers = [g for g in match.groups(
            ) if g and g.replace('.', '').isdigit()]
            if numbers:
                entities["price_max"] = float(numbers[0])
                break

    # ---------- Quantity Extraction ----------
    quantity_match = re.search(
        r"(\d+)\s+(products?|items?|orders?|results?)", text_l)
    if quantity_match:
        entities["quantity"] = int(quantity_match.group(1))

    # ---------- Delivery Delay ----------
    delay_match = re.search(r"(\d+)\s+days?", text_l)
    if delay_match:
        entities["delivery_delay"] = int(delay_match.group(1))

    # ---------- Sentiment ----------
    for s in ["positive", "negative", "neutral"]:
        if s in text_l:
            entities["sentiment"] = s
            break

    return entities


def get_db_categories(neo4j_conn) -> List[str]:
    """
    Fetch all unique categories from the database
    Should be called once during initialization
    """
    query = """
    MATCH (p:Product)
    RETURN DISTINCT p.product_category_name as category
    ORDER BY category
    """

    results = neo4j_conn.query(query)
    return [r['category'] for r in results if r['category']]


# ---------- Optional: Validation Function ----------
def validate_entities(entities: Dict, intent: str) -> bool:
    """
    Check if extracted entities are sufficient for the given intent
    Returns True if entities are adequate, False otherwise
    """
    # Different intents require different entities
    requirements = {
        "search": ["category"],  # At least category for good search
        "recommendation": ["category"],
        "reviews": ["category"],
        "ranking": ["category"],
        "delivery_analysis": [],  # No required entities
        "seller_analysis": [],
        "analytics": [],
        "qa": []
    }

    required = requirements.get(intent, [])

    # Check if at least one required entity is present
    if not required:
        return True  # No requirements

    return any(entities.get(req) is not None for req in required)


def get_db_categories(neo4j_conn) -> List[str]:
    """
    Fetch all unique categories from the database
    Should be called once during initialization
    """
    query = """
    MATCH (p:Product)
    RETURN DISTINCT p.product_category_name as category
    ORDER BY category
    """

    results = neo4j_conn.query(query)
    return [r['category'] for r in results if r['category']]


# ============================================================================
# PART 2.a: BASELINE - CYPHER QUERY TEMPLATES
# ============================================================================
"""
CORRECTED CYPHER QUERY LIBRARY
Fixed: OrderItem → Order_Item (with underscore)

Your schema:
- Nodes: Customer, Order, Order_Item, Product, Seller, Review, State
- Relationships: PLACED, CONTAINS, REFERS_TO, SOLD_BY, WROTE, REVIEWS, LOCATED_IN

Relationship patterns (most likely):
- Customer -[:PLACED]-> Order
- Order -[:CONTAINS]-> Order_Item
- Order_Item -[:REFERS_TO]-> Product
- Order_Item -[:SOLD_BY]-> Seller
- Review -[:REVIEWS]-> Order
- Customer -[:WROTE]-> Review
- Customer -[:LOCATED_IN]-> State
"""

"""
MEMORY-OPTIMIZED CYPHER QUERY LIBRARY
Fixes for Neo4j Aura FREE tier (250MB memory limit)

Key optimizations:
1. Reduced LIMIT on aggregations
2. Removed expensive multi-hop joins
3. Added LIMIT before aggregations
4. Simplified complex queries
"""


class CypherQueryLibrary:
    """Memory-optimized query library for Neo4j Aura FREE tier"""

    @staticmethod
    def get_query(intent: str, entities: Dict) -> Tuple[str, Dict]:
        """
        Select and parameterize appropriate Cypher query based on intent
        """

        # Query 1: Product search by category and rating (OPTIMIZED)
        if intent == "search" and entities.get("category") and entities.get("rating_min"):
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WHERE p.product_category_name = $category
            WITH p LIMIT 100
            MATCH (oi2:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi2)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WHERE r.review_score >= $rating_min
            WITH p, AVG(r.review_score) as avg_rating, COUNT(r) as review_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC, review_count DESC
            LIMIT 10
            """
            params = {
                "category": entities["category"],
                "rating_min": entities["rating_min"]
            }
            return query, params

        # Query 2: Products by location (SIMPLIFIED)
        elif intent == "search" and entities.get("category") and entities.get("city"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:PLACED]-(c:Customer)
            WHERE c.customer_city = $city
            WITH p, COUNT(DISTINCT o) as order_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   order_count,
                   $city as location
            ORDER BY order_count DESC
            LIMIT 10
            """
            params = {
                "city": entities["city"],
                "category": entities["category"]
            }
            return query, params

        # Query 3: Review retrieval for category (OPTIMIZED)
        elif intent == "reviews" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 20
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            RETURN r.review_id as review_id,
                   r.review_score as score,
                   r.review_comment_title as title,
                   r.review_comment_message as message,
                   p.product_category_name as category,
                   p.product_id as product_id
            ORDER BY r.review_creation_date DESC
            LIMIT 20
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 4: Delivery delay analysis (SAFE - no date calc)
        elif intent == "delivery_analysis":
            query = """
            MATCH (o:Order)-[:CONTAINS]->(oi:Order_Item)-[:REFERS_TO]->(p:Product)
            WHERE o.order_delivered_customer_date IS NOT NULL
            WITH p.product_category_name as category,
                 COUNT(DISTINCT o) as total_orders
            RETURN category, total_orders
            ORDER BY total_orders DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 5: Seller performance analysis (OPTIMIZED)
        elif intent == "seller_analysis":
            query = """
            MATCH (s:Seller)<-[:SOLD_BY]-(oi:Order_Item)
            WITH s, oi LIMIT 1000
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH s, AVG(r.review_score) as avg_rating, COUNT(DISTINCT oi) as total_sales
            WHERE total_sales >= 5
            RETURN s.seller_id as seller_id,
                   avg_rating,
                   total_sales
            ORDER BY avg_rating DESC, total_sales DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 6: Top products ranking (MEMORY OPTIMIZED!)
        elif intent == "ranking" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, oi LIMIT 200
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE review_count >= 3
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC
            LIMIT 10
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 7: Category analytics (SAFE)
        elif intent == "analytics" and not entities.get("category"):
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p.product_category_name as category,
                 COUNT(oi) as total_items,
                 AVG(oi.price) as avg_price
            RETURN category, total_items, avg_price
            ORDER BY total_items DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Query 8: Review scores by status (SAFE)
        elif intent == "analytics" and "impact" in str(entities.get("category", "")).lower():
            query = """
            MATCH (o:Order)<-[:REVIEWS]-(r:Review)
            WHERE o.order_status IS NOT NULL
            WITH o.order_status as status,
                 AVG(r.review_score) as avg_score,
                 COUNT(r) as review_count
            RETURN status, avg_score, review_count
            ORDER BY avg_score DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 9: State-based statistics (SIMPLIFIED)
        elif intent == "analytics" and entities.get("city"):
            query = """
            MATCH (c:Customer)
            WHERE c.customer_city = $city
            WITH c LIMIT 100
            MATCH (c)-[:PLACED]->(o:Order)
            MATCH (c)-[:LOCATED_IN]->(s:State)
            WITH s.state as state,
                 COUNT(DISTINCT c) as customer_count,
                 COUNT(DISTINCT o) as order_count
            RETURN state, customer_count, order_count
            """
            params = {"city": entities["city"]}
            return query, params

        # Query 10: Product recommendations (OPTIMIZED)
        elif intent == "recommendation" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, oi LIMIT 200
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE avg_rating >= 4.0 AND review_count >= 5
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC
            LIMIT 10
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 11: General QA (SAFE)
        elif intent == "qa":
            query = """
            MATCH (p:Product)
            WITH p.product_category_name as category, COUNT(p) as product_count
            RETURN category, product_count
            ORDER BY product_count DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Query 12: Simple category search (SAFE)
        elif intent == "search" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, COUNT(oi) as order_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   order_count
            ORDER BY order_count DESC
            LIMIT 20
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 13: Products with most reviews (SIMPLIFIED)
        elif intent == "ranking":
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p, oi LIMIT 500
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE review_count >= 3
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY review_count DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 14: Top selling categories (SAFE)
        elif intent == "analytics":
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p.product_category_name as category,
                 COUNT(oi) as items_sold
            RETURN category, items_sold
            ORDER BY items_sold DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Default fallback (SAFE)
        else:
            query = """
            MATCH (p:Product)
            RETURN p.product_id as product_id,
                   p.product_category_name as category
            LIMIT 20
            """
            params = {}
            return query, params


# ============================================================================
# PART 2.b: EMBEDDINGS - NODE & FEATURE VECTOR EMBEDDINGS
# ============================================================================

class EmbeddingRetrieval:
    """Handles embedding-based retrieval from Neo4j vector index"""

    def __init__(self, neo4j_conn, embedder: InputEmbedder):
        """
        Initialize embedding retrieval
        Args:
            neo4j_conn: Neo4jConnection instance
            embedder: InputEmbedder instance
        """
        self.conn = neo4j_conn
        self.embedder = embedder

    def check_database_size(self):
        """Check how many nodes exist (diagnostic function)"""
        query = """
        MATCH (p:Product)
        RETURN count(p) as total_products,
               count(DISTINCT p.product_category_name) as unique_categories
        """
        result = self.conn.query(query)
        if result:
            print(f"Database contains:")
            print(f"  Total Products: {result[0]['total_products']:,}")
            print(f"  Unique Categories: {result[0]['unique_categories']}")
        return result

    def create_node_embeddings(self, node_label='Product',
                               text_property='product_category_name',
                               batch_size=500):
        """
        Create and store node embeddings in Neo4j (OPTIMIZED VERSION)
        This should be run once to initialize embeddings
        """
        print(f"Fetching {node_label} nodes...")

        # Fetch all UNIQUE categories (much faster than all nodes)
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{text_property} IS NOT NULL
        RETURN DISTINCT n.{text_property} as text
        """
        results = self.conn.query(query)

        # Generate embeddings for unique categories only
        unique_texts = [r['text'] for r in results]
        print(f"Found {len(unique_texts)} unique categories")
        print(f"Generating embeddings...")

        embeddings = self.embedder.embed_batch(unique_texts)

        # Create a map of category -> embedding
        embedding_map = {text: emb.tolist()
                         for text, emb in zip(unique_texts, embeddings)}

        # Batch update ALL nodes with same category in one query
        print(f"Updating nodes in database...")
        for text, embedding in embedding_map.items():
            update_query = f"""
            MATCH (n:{node_label})
            WHERE n.{text_property} = $text
            SET n.embedding = $embedding
            """
            self.conn.query(update_query, {
                'text': text,
                'embedding': embedding
            })
            print(f"  ✓ Updated all nodes with category: {text}")

        print(
            f"✓ Created embeddings for {len(unique_texts)} unique categories")

    def create_feature_vector_embeddings(self, batch_size=1000,
                                         limit=None,
                                         include_category=False):
        """
        Create feature vector embeddings combining multiple properties (OPTIMIZED)
        Args:
            batch_size: Number of products to process at once
            limit: Optional limit on total products
            include_category: If False, excludes category to avoid data leakage
        """
        print("Fetching product features...")

        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        MATCH (p:Product)
        RETURN p.product_id as product_id,
               p.product_category_name as category,
               p.product_description_lenght as desc_length,
               p.product_photos_qty as photos
        {limit_clause}
        """
        results = self.conn.query(query)

        print(f"Processing {len(results)} products...")

        # Process in batches
        total_processed = 0
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]

            # Create text descriptions from features
            feature_texts = []
            product_ids = []

            for r in batch:
                # Option to exclude category to avoid data leakage
                if include_category:
                    text = f"Product category: {r['category']}, Description length: {r['desc_length']}, Photos: {r['photos']}"
                else:
                    text = f"Description length: {r['desc_length']}, Number of photos: {r['photos']}"

                feature_texts.append(text)
                product_ids.append(r['product_id'])

            # Generate embeddings for batch
            print(f"  Embedding batch {i//batch_size + 1}...")
            embeddings = self.embedder.embed_batch(feature_texts)

            # Batch update in Neo4j using UNWIND
            print(f"  Updating database...")
            update_query = """
            UNWIND $data as row
            MATCH (p:Product {product_id: row.product_id})
            SET p.feature_embedding = row.embedding
            """

            data = [
                {'product_id': pid, 'embedding': emb.tolist()}
                for pid, emb in zip(product_ids, embeddings)
            ]

            self.conn.query(update_query, {'data': data})
            total_processed += len(batch)
            print(f"  ✓ Processed {total_processed}/{len(results)} products")

        print(f"✓ Created feature embeddings for {len(results)} products")

    def create_vector_index(self, index_name='product_embeddings',
                            node_label='Product',
                            embedding_property='embedding'):
        """Create vector index in Neo4j for fast similarity search"""
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON n.{embedding_property}
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {self.embedder.model.get_sentence_embedding_dimension()},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            self.conn.query(query)
            print(f"✓ Created vector index: {index_name}")
        except Exception as e:
            print(f"Note: Vector index may already exist: {e}")

    def semantic_search(self, query_text: str, top_k=10,
                        index_name='product_embeddings'):
        """
        Perform semantic similarity search
        Args:
            query_text: User's input text
            top_k: Number of results to return
            index_name: Name of vector index to use
        Returns:
            List of similar nodes with scores
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query_text)

        # Search using vector index
        search_query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node.product_id as product_id,
               node.product_category_name as category,
               score
        ORDER BY score DESC
        """

        try:
            results = self.conn.query(search_query, {
                'index_name': index_name,
                'top_k': top_k,
                'query_vector': query_embedding.tolist()
            })
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []


# ============================================================================
# INTEGRATED RETRIEVAL SYSTEM (YOUR retrieve() FUNCTION INTEGRATED)
# ============================================================================

class GraphRAGRetriever:
    """Main retrieval system combining baseline and embeddings"""

    def __init__(self, neo4j_conn, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize Graph-RAG retriever
        Args:
            neo4j_conn: Neo4jConnection instance
            embedding_model: Name of embedding model to use
        """
        self.conn = neo4j_conn
        self.cypher_lib = CypherQueryLibrary()
        self.embedder = InputEmbedder(embedding_model)
        self.embedding_retrieval = EmbeddingRetrieval(
            neo4j_conn, self.embedder)

    def retrieve_baseline(self, intent: str, entities: Dict) -> Dict:
        """
        Baseline retrieval using Cypher queries
        Args:
            intent: Classified intent
            entities: Extracted entities
        Returns:
            dict with query, params, and results
        """
        query, params = self.cypher_lib.get_query(intent, entities)
        results = self.conn.query(query, params)

        return {
            'method': 'baseline',
            'query': query,
            'params': params,
            'results': results,
            'count': len(results) if results else 0
        }

    def retrieve_embeddings(self, query_text: str, top_k=10) -> Dict:
        """
        Embedding-based retrieval using semantic search
        Args:
            query_text: Original user query
            top_k: Number of results
        Returns:
            dict with results
        """
        results = self.embedding_retrieval.semantic_search(query_text, top_k)

        return {
            'method': 'embeddings',
            'query_text': query_text,
            'results': results,
            'count': len(results) if results else 0
        }

    # ========================================================================
    # YOUR retrieve() FUNCTION - THIS IS THE MAIN ENTRY POINT
    # ========================================================================

    def retrieve(self, query_text: str, intent: Optional[str],
                 entities: Dict, use_embeddings=True) -> Dict:
        """
        Main retrieval function with intelligent fallback logic

        Args:
            query_text: Original user query
            intent: Classified intent (None if classification failed)
            entities: Extracted entities
            use_embeddings: Whether to use embedding fallback

        Returns:
            dict with retrieval results and metadata
        """
        # Case 1: Intent failed → embeddings
        if intent is None:
            print("⚠️ Intent classification failed → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'intent_failure'
            return result

        # Case 2: Category missing → embeddings fallback
        if entities.get("category") is None and use_embeddings:
            print("⚠️ No category extracted → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'missing_category'
            return result

        # Case 3: Try baseline first
        print(f"✓ Using baseline retrieval for intent: {intent}")
        baseline = self.retrieve_baseline(intent, entities)

        # If baseline returns empty → embeddings fallback
        if baseline["count"] == 0 and use_embeddings:
            print("⚠️ Baseline returned no results → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'empty_results'
            result['attempted_baseline'] = baseline
            return result

        # Baseline succeeded
        return baseline

    def retrieve_hybrid(self, query_text: str, intent: Optional[str],
                        entities: Dict, top_k=10) -> Dict:
        """
        Combined retrieval: baseline + embeddings
        Useful for comprehensive results or comparison

        Args:
            query_text: Original user query
            intent: Classified intent
            entities: Extracted entities
            top_k: Number of embedding results

        Returns:
            Combined results from both methods
        """
        baseline_results = self.retrieve_baseline(
            intent, entities) if intent else None
        embedding_results = self.retrieve_embeddings(query_text, top_k)

        # Combine results
        combined = {
            'method': 'hybrid',
            'baseline': baseline_results,
            'embeddings': embedding_results,
            'baseline_count': baseline_results['count'] if baseline_results else 0,
            'embeddings_count': embedding_results['count'],
            'total_count': (baseline_results['count'] if baseline_results else 0) + embedding_results['count']
        }

        return combined


# ============================================================================
# PART 2.a: BASELINE - CYPHER QUERY TEMPLATES
# ============================================================================

"""
CORRECTED CYPHER QUERY LIBRARY
Fixed: OrderItem → Order_Item (with underscore)

Your schema:
- Nodes: Customer, Order, Order_Item, Product, Seller, Review, State
- Relationships: PLACED, CONTAINS, REFERS_TO, SOLD_BY, WROTE, REVIEWS, LOCATED_IN

Relationship patterns (most likely):
- Customer -[:PLACED]-> Order
- Order -[:CONTAINS]-> Order_Item
- Order_Item -[:REFERS_TO]-> Product
- Order_Item -[:SOLD_BY]-> Seller
- Review -[:REVIEWS]-> Order
- Customer -[:WROTE]-> Review
- Customer -[:LOCATED_IN]-> State
"""

"""
MEMORY-OPTIMIZED CYPHER QUERY LIBRARY
Fixes for Neo4j Aura FREE tier (250MB memory limit)

Key optimizations:
1. Reduced LIMIT on aggregations
2. Removed expensive multi-hop joins
3. Added LIMIT before aggregations
4. Simplified complex queries
"""


class CypherQueryLibrary:
    """Memory-optimized query library for Neo4j Aura FREE tier"""

    @staticmethod
    def get_query(intent: str, entities: Dict) -> Tuple[str, Dict]:
        """
        Select and parameterize appropriate Cypher query based on intent
        """

        # Query 1: Product search by category and rating (OPTIMIZED)
        if intent == "search" and entities.get("category") and entities.get("rating_min"):
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WHERE p.product_category_name = $category
            WITH p LIMIT 100
            MATCH (oi2:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi2)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WHERE r.review_score >= $rating_min
            WITH p, AVG(r.review_score) as avg_rating, COUNT(r) as review_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC, review_count DESC
            LIMIT 10
            """
            params = {
                "category": entities["category"],
                "rating_min": entities["rating_min"]
            }
            return query, params

        # Query 2: Products by location (SIMPLIFIED)
        elif intent == "search" and entities.get("category") and entities.get("city"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:PLACED]-(c:Customer)
            WHERE c.customer_city = $city
            WITH p, COUNT(DISTINCT o) as order_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   order_count,
                   $city as location
            ORDER BY order_count DESC
            LIMIT 10
            """
            params = {
                "city": entities["city"],
                "category": entities["category"]
            }
            return query, params

        # Query 3: Review retrieval for category (OPTIMIZED)
        elif intent == "reviews" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 20
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            RETURN r.review_id as review_id,
                   r.review_score as score,
                   r.review_comment_title as title,
                   r.review_comment_message as message,
                   p.product_category_name as category,
                   p.product_id as product_id
            ORDER BY r.review_creation_date DESC
            LIMIT 20
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 4: Delivery delay analysis (SAFE - no date calc)
        elif intent == "delivery_analysis":
            query = """
            MATCH (o:Order)-[:CONTAINS]->(oi:Order_Item)-[:REFERS_TO]->(p:Product)
            WHERE o.order_delivered_customer_date IS NOT NULL
            WITH p.product_category_name as category,
                 COUNT(DISTINCT o) as total_orders
            RETURN category, total_orders
            ORDER BY total_orders DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 5: Seller performance analysis (OPTIMIZED)
        elif intent == "seller_analysis":
            query = """
            MATCH (s:Seller)<-[:SOLD_BY]-(oi:Order_Item)
            WITH s, oi LIMIT 1000
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH s, AVG(r.review_score) as avg_rating, COUNT(DISTINCT oi) as total_sales
            WHERE total_sales >= 5
            RETURN s.seller_id as seller_id,
                   avg_rating,
                   total_sales
            ORDER BY avg_rating DESC, total_sales DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 6: Top products ranking (MEMORY OPTIMIZED!)
        elif intent == "ranking" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, oi LIMIT 200
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE review_count >= 3
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC
            LIMIT 10
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 7: Category analytics (SAFE)
        elif intent == "analytics" and not entities.get("category"):
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p.product_category_name as category,
                 COUNT(oi) as total_items,
                 AVG(oi.price) as avg_price
            RETURN category, total_items, avg_price
            ORDER BY total_items DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Query 8: Review scores by status (SAFE)
        elif intent == "analytics" and "impact" in str(entities.get("category", "")).lower():
            query = """
            MATCH (o:Order)<-[:REVIEWS]-(r:Review)
            WHERE o.order_status IS NOT NULL
            WITH o.order_status as status,
                 AVG(r.review_score) as avg_score,
                 COUNT(r) as review_count
            RETURN status, avg_score, review_count
            ORDER BY avg_score DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 9: State-based statistics (SIMPLIFIED)
        elif intent == "analytics" and entities.get("city"):
            query = """
            MATCH (c:Customer)
            WHERE c.customer_city = $city
            WITH c LIMIT 100
            MATCH (c)-[:PLACED]->(o:Order)
            MATCH (c)-[:LOCATED_IN]->(s:State)
            WITH s.state as state,
                 COUNT(DISTINCT c) as customer_count,
                 COUNT(DISTINCT o) as order_count
            RETURN state, customer_count, order_count
            """
            params = {"city": entities["city"]}
            return query, params

        # Query 10: Product recommendations (OPTIMIZED)
        elif intent == "recommendation" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, oi LIMIT 200
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE avg_rating >= 4.0 AND review_count >= 5
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY avg_rating DESC
            LIMIT 10
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 11: General QA (SAFE)
        elif intent == "qa":
            query = """
            MATCH (p:Product)
            WITH p.product_category_name as category, COUNT(p) as product_count
            RETURN category, product_count
            ORDER BY product_count DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Query 12: Simple category search (SAFE)
        elif intent == "search" and entities.get("category"):
            query = """
            MATCH (p:Product)
            WHERE p.product_category_name = $category
            WITH p LIMIT 50
            MATCH (oi:Order_Item)-[:REFERS_TO]->(p)
            WITH p, COUNT(oi) as order_count
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   order_count
            ORDER BY order_count DESC
            LIMIT 20
            """
            params = {"category": entities["category"]}
            return query, params

        # Query 13: Products with most reviews (SIMPLIFIED)
        elif intent == "ranking":
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p, oi LIMIT 500
            MATCH (oi)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
            WITH p,
                 AVG(r.review_score) as avg_rating,
                 COUNT(r) as review_count
            WHERE review_count >= 3
            RETURN p.product_id as product_id,
                   p.product_category_name as category,
                   avg_rating,
                   review_count
            ORDER BY review_count DESC
            LIMIT 10
            """
            params = {}
            return query, params

        # Query 14: Top selling categories (SAFE)
        elif intent == "analytics":
            query = """
            MATCH (p:Product)<-[:REFERS_TO]-(oi:Order_Item)
            WITH p.product_category_name as category,
                 COUNT(oi) as items_sold
            RETURN category, items_sold
            ORDER BY items_sold DESC
            LIMIT 15
            """
            params = {}
            return query, params

        # Default fallback (SAFE)
        else:
            query = """
            MATCH (p:Product)
            RETURN p.product_id as product_id,
                   p.product_category_name as category
            LIMIT 20
            """
            params = {}
            return query, params


# ============================================================================
# PART 2.b: EMBEDDINGS - NODE & FEATURE VECTOR EMBEDDINGS
# ============================================================================

class EmbeddingRetrieval:
    """Handles embedding-based retrieval from Neo4j vector index"""

    def __init__(self, neo4j_conn, embedder: InputEmbedder):
        """
        Initialize embedding retrieval
        Args:
            neo4j_conn: Neo4jConnection instance
            embedder: InputEmbedder instance
        """
        self.conn = neo4j_conn
        self.embedder = embedder

    def check_database_size(self):
        """Check how many nodes exist (diagnostic function)"""
        query = """
        MATCH (p:Product)
        RETURN count(p) as total_products,
               count(DISTINCT p.product_category_name) as unique_categories
        """
        result = self.conn.query(query)
        if result:
            print(f"Database contains:")
            print(f"  Total Products: {result[0]['total_products']:,}")
            print(f"  Unique Categories: {result[0]['unique_categories']}")
        return result

    def create_node_embeddings(self, node_label='Product',
                               text_property='product_category_name',
                               batch_size=500):
        """
        Create and store node embeddings in Neo4j (OPTIMIZED VERSION)
        This should be run once to initialize embeddings
        """
        print(f"Fetching {node_label} nodes...")

        # Fetch all UNIQUE categories (much faster than all nodes)
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{text_property} IS NOT NULL
        RETURN DISTINCT n.{text_property} as text
        """
        results = self.conn.query(query)

        # Generate embeddings for unique categories only
        unique_texts = [r['text'] for r in results]
        print(f"Found {len(unique_texts)} unique categories")
        print(f"Generating embeddings...")

        embeddings = self.embedder.embed_batch(unique_texts)

        # Create a map of category -> embedding
        embedding_map = {text: emb.tolist()
                         for text, emb in zip(unique_texts, embeddings)}

        # Batch update ALL nodes with same category in one query
        print(f"Updating nodes in database...")
        for text, embedding in embedding_map.items():
            update_query = f"""
            MATCH (n:{node_label})
            WHERE n.{text_property} = $text
            SET n.embedding = $embedding
            """
            self.conn.query(update_query, {
                'text': text,
                'embedding': embedding
            })
            print(f"  ✓ Updated all nodes with category: {text}")

        print(
            f"✓ Created embeddings for {len(unique_texts)} unique categories")

    def create_feature_vector_embeddings(self, batch_size=1000,
                                         limit=None,
                                         include_category=False):
        """
        Create feature vector embeddings combining multiple properties (OPTIMIZED)
        Args:
            batch_size: Number of products to process at once
            limit: Optional limit on total products
            include_category: If False, excludes category to avoid data leakage
        """
        print("Fetching product features...")

        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        MATCH (p:Product)
        RETURN p.product_id as product_id,
               p.product_category_name as category,
               p.product_description_lenght as desc_length,
               p.product_photos_qty as photos
        {limit_clause}
        """
        results = self.conn.query(query)

        print(f"Processing {len(results)} products...")

        # Process in batches
        total_processed = 0
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]

            # Create text descriptions from features
            feature_texts = []
            product_ids = []

            for r in batch:
                # Option to exclude category to avoid data leakage
                if include_category:
                    text = f"Product category: {r['category']}, Description length: {r['desc_length']}, Photos: {r['photos']}"
                else:
                    text = f"Description length: {r['desc_length']}, Number of photos: {r['photos']}"

                feature_texts.append(text)
                product_ids.append(r['product_id'])

            # Generate embeddings for batch
            print(f"  Embedding batch {i//batch_size + 1}...")
            embeddings = self.embedder.embed_batch(feature_texts)

            # Batch update in Neo4j using UNWIND
            print(f"  Updating database...")
            update_query = """
            UNWIND $data as row
            MATCH (p:Product {product_id: row.product_id})
            SET p.feature_embedding = row.embedding
            """

            data = [
                {'product_id': pid, 'embedding': emb.tolist()}
                for pid, emb in zip(product_ids, embeddings)
            ]

            self.conn.query(update_query, {'data': data})
            total_processed += len(batch)
            print(f"  ✓ Processed {total_processed}/{len(results)} products")

        print(f"✓ Created feature embeddings for {len(results)} products")

    def create_vector_index(self, index_name='product_embeddings',
                            node_label='Product',
                            embedding_property='embedding'):
        """Create vector index in Neo4j for fast similarity search"""
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON n.{embedding_property}
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {self.embedder.model.get_sentence_embedding_dimension()},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            self.conn.query(query)
            print(f"✓ Created vector index: {index_name}")
        except Exception as e:
            print(f"Note: Vector index may already exist: {e}")

    def semantic_search(self, query_text: str, top_k=10,
                        index_name='product_embeddings'):
        """
        Perform semantic similarity search
        Args:
            query_text: User's input text
            top_k: Number of results to return
            index_name: Name of vector index to use
        Returns:
            List of similar nodes with scores
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query_text)

        # Search using vector index
        search_query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node.product_id as product_id,
               node.product_category_name as category,
               score
        ORDER BY score DESC
        """

        try:
            results = self.conn.query(search_query, {
                'index_name': index_name,
                'top_k': top_k,
                'query_vector': query_embedding.tolist()
            })
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []


# ============================================================================
# INTEGRATED RETRIEVAL SYSTEM (YOUR retrieve() FUNCTION INTEGRATED)
# ============================================================================

class GraphRAGRetriever:
    """Main retrieval system combining baseline and embeddings"""

    def __init__(self, neo4j_conn, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize Graph-RAG retriever
        Args:
            neo4j_conn: Neo4jConnection instance
            embedding_model: Name of embedding model to use
        """
        self.conn = neo4j_conn
        self.cypher_lib = CypherQueryLibrary()
        self.embedder = InputEmbedder(embedding_model)
        self.embedding_retrieval = EmbeddingRetrieval(
            neo4j_conn, self.embedder)

    def retrieve_baseline(self, intent: str, entities: Dict) -> Dict:
        """
        Baseline retrieval using Cypher queries
        Args:
            intent: Classified intent
            entities: Extracted entities
        Returns:
            dict with query, params, and results
        """
        query, params = self.cypher_lib.get_query(intent, entities)
        results = self.conn.query(query, params)

        return {
            'method': 'baseline',
            'query': query,
            'params': params,
            'results': results,
            'count': len(results) if results else 0
        }

    def retrieve_embeddings(self, query_text: str, top_k=10) -> Dict:
        """
        Embedding-based retrieval using semantic search
        Args:
            query_text: Original user query
            top_k: Number of results
        Returns:
            dict with results
        """
        results = self.embedding_retrieval.semantic_search(query_text, top_k)

        return {
            'method': 'embeddings',
            'query_text': query_text,
            'results': results,
            'count': len(results) if results else 0
        }

    # ========================================================================
    # YOUR retrieve() FUNCTION - THIS IS THE MAIN ENTRY POINT
    # ========================================================================

    def retrieve(self, query_text: str, intent: Optional[str],
                 entities: Dict, use_embeddings=True) -> Dict:
        """
        Main retrieval function with intelligent fallback logic

        Args:
            query_text: Original user query
            intent: Classified intent (None if classification failed)
            entities: Extracted entities
            use_embeddings: Whether to use embedding fallback

        Returns:
            dict with retrieval results and metadata
        """
        # Case 1: Intent failed → embeddings
        if intent is None:
            print("⚠️ Intent classification failed → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'intent_failure'
            return result

        # Case 2: Category missing → embeddings fallback
        if entities.get("category") is None and use_embeddings:
            print("⚠️ No category extracted → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'missing_category'
            return result

        # Case 3: Try baseline first
        print(f"✓ Using baseline retrieval for intent: {intent}")
        baseline = self.retrieve_baseline(intent, entities)

        # If baseline returns empty → embeddings fallback
        if baseline["count"] == 0 and use_embeddings:
            print("⚠️ Baseline returned no results → Using embeddings")
            result = self.retrieve_embeddings(query_text)
            result['fallback_reason'] = 'empty_results'
            result['attempted_baseline'] = baseline
            return result

        # Baseline succeeded
        return baseline

    def retrieve_hybrid(self, query_text: str, intent: Optional[str],
                        entities: Dict, top_k=10) -> Dict:
        """
        Combined retrieval: baseline + embeddings
        Useful for comprehensive results or comparison

        Args:
            query_text: Original user query
            intent: Classified intent
            entities: Extracted entities
            top_k: Number of embedding results

        Returns:
            Combined results from both methods
        """
        baseline_results = self.retrieve_baseline(
            intent, entities) if intent else None
        embedding_results = self.retrieve_embeddings(query_text, top_k)

        # Combine results
        combined = {
            'method': 'hybrid',
            'baseline': baseline_results,
            'embeddings': embedding_results,
            'baseline_count': baseline_results['count'] if baseline_results else 0,
            'embeddings_count': embedding_results['count'],
            'total_count': (baseline_results['count'] if baseline_results else 0) + embedding_results['count']
        }

        return combined

# ---------- Optional: Validation Function ----------


def validate_entities(entities: Dict, intent: str) -> bool:
    """
    Check if extracted entities are sufficient for the given intent
    Returns True if entities are adequate, False otherwise
    """
    # Different intents require different entities
    requirements = {
        "search": ["category"],  # At least category for good search
        "recommendation": ["category"],
        "reviews": ["category"],
        "ranking": ["category"],
        "delivery_analysis": [],  # No required entities
        "seller_analysis": [],
        "analytics": [],
        "qa": []
    }

    required = requirements.get(intent, [])

    # Check if at least one required entity is present
    if not required:
        return True  # No requirements

    return any(entities.get(req) is not None for req in required)


_pipeline = None


def get_pipeline(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    hf_token,
    llm_model="mistral-7b"
):
    global _pipeline

    if _pipeline is None:
        conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)

        retriever = GraphRAGRetriever(conn, embedding_model="all-MiniLM-L6-v2")
        setup = GraphRAGSetup(retriever)
        setup.initialize_embeddings(skip_if_exists=True)

        _pipeline = GraphRAGPipeline(
            conn,
            embedding_model="all-MiniLM-L6-v2",
            llm_model=llm_model,
            hf_token=hf_token
        )

    return _pipeline


@st.cache_resource
def load_pipeline(model_name):
    return get_pipeline(
        neo4j_uri="neo4j+s://df998545.databases.neo4j.io",
        neo4j_user="neo4j",
        neo4j_password="P49OO43v3OLd-Gu3PieJtAQw1ifPxlyxU9JUMq7H5MY",
        hf_token="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",
        llm_model=model_name
    )


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
st.markdown('<h1 class="main-header">🛍️ E-Commerce Graph-RAG Assistant</h1>',
            unsafe_allow_html=True)
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
selected_retrieval = st.sidebar.selectbox(
    "Choose Retrieval", list(retrieval_modes.keys()))

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

    pipeline = load_pipeline(llm_models[selected_model])

    with st.spinner("🔍 Running Graph-RAG pipeline..."):
        result = pipeline.process_query(
            user_query,
            retrieval_mode=retrieval_modes[selected_retrieval],
            verbose=False
        )

    # =============================
    # QUERY ANALYSIS
    # =============================
    st.markdown('<h2 class="section-header">📊 Query Analysis</h2>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Intent", result["intent"])
    with col2:
        st.metric("Retrieval", result["retrieval"]["method"])
    with col3:
        st.metric("Results", result["retrieval"]["count"])

    # =============================
    # CYPHER QUERY
    # =============================
    if show_cypher:
        st.markdown("### 🧠 Cypher Query")
        st.code(result["retrieval"].get(
            "cypher", "Generated internally"), language="cypher")

    # =============================
    # CONTEXT
    # =============================
    if show_context:
        st.markdown("### 📚 Knowledge Graph Context")
        st.text(result["context"])

    # =============================
    # AI ANSWER (FULL, REAL ANSWER)
    # =============================
    st.markdown('<h2 class="section-header">🤖 AI Answer</h2>',
                unsafe_allow_html=True)
    st.success(result["final_answer"])

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
<small>Powered by Neo4j + HuggingFace | Milestone 3</small>
</div>
""", unsafe_allow_html=True)
