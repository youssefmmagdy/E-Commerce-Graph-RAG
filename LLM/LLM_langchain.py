"""
LangChain-based LLM module for E-Commerce Knowledge Graph RAG system.
This is an alternative implementation using LangChain instead of direct HuggingFace InferenceClient.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


HF_TOKEN = os.getenv('HF_TOKEN')

# =============================================================================
# MODEL CONFIGURATIONS - 3 Models for Comparison
# =============================================================================
MODELS = {
    "gemma": {
        "name": "Gemma 2 2B",
        "model_id": "google/gemma-2-2b-it",
        "description": "Google's efficient 2B parameter model",
    },
    "llama_3b": {
        "name": "Llama 3.2 3B",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Meta's compact 3B instruction-tuned model",
    },
    "llama_1b": {
        "name": "Llama 3.2 1B",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "description": "Meta's lightweight 1B instruction-tuned model",
    },
}

# Create InferenceClients for each model (uses chat_completion which is conversational)
_clients: Dict[str, InferenceClient] = {}

def get_client(model_key: str) -> InferenceClient:
    """Get or create a HuggingFace InferenceClient for the specified model"""
    if model_key not in MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(MODELS.keys())}")
    
    if model_key not in _clients:
        _clients[model_key] = InferenceClient(
            model=MODELS[model_key]["model_id"],
            token=HF_TOKEN
        )
    return _clients[model_key]


# =============================================================================
# LANGCHAIN-COMPATIBLE LLM WRAPPER
# =============================================================================
class HuggingFaceChatLLM:
    """Custom LLM wrapper that uses InferenceClient with chat_completion (conversational task)"""
    
    def __init__(self, model_key: str, max_tokens: int = 500, temperature: float = 0.2):
        self.model_key = model_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = get_client(model_key)
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt and return the response"""
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


def get_llm(model_key: str, max_tokens: int = 500, temperature: float = 0.2) -> HuggingFaceChatLLM:
    """Create a LangChain-compatible LLM for the specified model"""
    if model_key not in MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(MODELS.keys())}")
    
    return HuggingFaceChatLLM(model_key, max_tokens, temperature)


# Create lazy-loaded LLM instances
_llm_cache: Dict[str, HuggingFaceChatLLM] = {}

def get_cached_llm(model_key: str, max_tokens: int = 500) -> HuggingFaceChatLLM:
    """Get or create a cached LLM instance"""
    cache_key = f"{model_key}_{max_tokens}"
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = get_llm(model_key, max_tokens)
    return _llm_cache[cache_key]


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================
@dataclass
class ModelResponse:
    """Stores a single model's response with metrics"""
    model_key: str
    model_name: str
    response: str
    response_time_ms: float
    input_tokens: int  # estimated
    output_tokens: int  # estimated
    success: bool
    error: str = ""


@dataclass 
class ComparisonResult:
    """Stores comparison results across all models"""
    query: str
    context_summary: str
    responses: List[ModelResponse]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "context_summary": self.context_summary,
            "responses": [
                {
                    "model": r.model_name,
                    "response": r.response,
                    "response_time_ms": r.response_time_ms,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.responses
            ]
        }


# =============================================================================
# LANGCHAIN PROMPT TEMPLATES
# =============================================================================
RAG_PROMPT_TEMPLATE = """PERSONA:
You are an intelligent E-Commerce marketplace assistant with expertise in:
- Analyzing product information and categories
- Understanding customer preferences
- Recommending products based on features
- Providing helpful marketplace insights

You are knowledgeable, helpful, and always base your answers on the provided information.

RETRIEVED KNOWLEDGE GRAPH DATA:
============================================================

{context}

TASK:
Answer the following user question using ONLY the provided knowledge graph data above:

Question: "{question}"

Requirements:
1. Base your answer exclusively on the retrieved product information
2. Reference specific products and their features when relevant
3. Be concise and direct
4. If the information cannot be answered from the context, clearly state this
5. Do not make assumptions or hallucinate information

ANSWER:"""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def estimate_tokens(text: str) -> int:
    """Rough token estimation (avg 4 chars per token)"""
    return len(text) // 4


def build_context(combined_records: List, max_records: int = 5) -> str:
    """Build context string from combined records"""
    
    # Handle empty records
    if not combined_records:
        return "No records found for this query.\n"
    
    context = ""
    for i, item in enumerate(combined_records[:max_records], 1):
        context += f"Record {i}:\n"
        
        try:
            if hasattr(item, 'keys'):
                for key in item.keys():
                    value = item[key]
                    # Convert Neo4j Node objects to string
                    if hasattr(value, 'items'):
                        value = dict(value.items())
                    # Truncate very long values
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:500] + "..."
                    context += f"  - {key}: {str_value}\n"
            elif isinstance(item, dict):
                for key, value in item.items():
                    # Convert Neo4j Node objects to string
                    if hasattr(value, 'items'):
                        value = dict(value.items())
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:500] + "..."
                    context += f"  - {key}: {str_value}\n"
            elif isinstance(item, str):
                context += f"  - Data: {item[:500] if len(item) > 500 else item}\n"
        except Exception as e:
            context += f"  - Error reading record: {str(e)}\n"
        
        context += "\n"
    
    return context


def build_prompt(context: str, user_prompt: str) -> str:
    """Build the full prompt with persona, context, and task (for backward compatibility)"""
    return rag_prompt.format(context=context, question=user_prompt)


# =============================================================================
# LANGCHAIN CHAIN BUILDERS
# =============================================================================
def create_rag_chain(model_key: str = "gemma", max_tokens: int = 500):
    """Create a LangChain RAG chain for the specified model"""
    llm = get_cached_llm(model_key, max_tokens)
    
    def invoke_llm(prompt_value) -> str:
        # Convert StringPromptValue to string
        prompt_str = prompt_value.to_string() if hasattr(prompt_value, 'to_string') else str(prompt_value)
        return llm.invoke(prompt_str)
    
    chain = (
        rag_prompt 
        | RunnableLambda(invoke_llm)
    )
    
    return chain


def create_full_rag_chain(model_key: str = "gemma", max_tokens: int = 500):
    """
    Create a full RAG chain that accepts records and question.
    Returns a chain that can be invoked with {"records": [...], "question": "..."}
    """
    llm = get_cached_llm(model_key, max_tokens)
    
    def format_records(inputs):
        return {
            "context": build_context(inputs["records"]),
            "question": inputs["question"]
        }
    
    def invoke_llm(prompt_value) -> str:
        # Convert StringPromptValue to string
        prompt_str = prompt_value.to_string() if hasattr(prompt_value, 'to_string') else str(prompt_value)
        return llm.invoke(prompt_str)
    
    chain = (
        RunnablePassthrough()
        | RunnableLambda(format_records)
        | rag_prompt
        | RunnableLambda(invoke_llm)
    )
    
    return chain


# =============================================================================
# SINGLE MODEL CALL
# =============================================================================
def call_model(model_key: str, prompt: str, max_tokens: int = 500) -> ModelResponse:
    """Call a specific model and return response with metrics"""
    if model_key not in MODELS:
        return ModelResponse(
            model_key=model_key,
            model_name="Unknown",
            response="",
            response_time_ms=0,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=f"Model '{model_key}' not found"
        )
    
    model_name = MODELS[model_key]["name"]
    input_tokens = estimate_tokens(prompt)
    
    start_time = time.time()
    try:
        llm = get_cached_llm(model_key, max_tokens)
        response_text = llm.invoke(prompt)
        end_time = time.time()
        
        output_tokens = estimate_tokens(response_text)
        
        return ModelResponse(
            model_key=model_key,
            model_name=model_name,
            response=response_text,
            response_time_ms=(end_time - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True
        )
    except Exception as e:
        end_time = time.time()
        return ModelResponse(
            model_key=model_key,
            model_name=model_name,
            response="",
            response_time_ms=(end_time - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=0,
            success=False,
            error=str(e)
        )


def call_model_with_chain(model_key: str, records: List, question: str, max_tokens: int = 500) -> ModelResponse:
    """Call a model using the LangChain RAG chain"""
    if model_key not in MODELS:
        return ModelResponse(
            model_key=model_key,
            model_name="Unknown",
            response="",
            response_time_ms=0,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=f"Model '{model_key}' not found"
        )
    
    model_name = MODELS[model_key]["name"]
    context = build_context(records)
    input_tokens = estimate_tokens(context + question)
    
    start_time = time.time()
    try:
        chain = create_rag_chain(model_key, max_tokens)
        response_text = chain.invoke({"context": context, "question": question})
        end_time = time.time()
        
        output_tokens = estimate_tokens(response_text)
        
        return ModelResponse(
            model_key=model_key,
            model_name=model_name,
            response=response_text,
            response_time_ms=(end_time - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True
        )
    except Exception as e:
        end_time = time.time()
        return ModelResponse(
            model_key=model_key,
            model_name=model_name,
            response="",
            response_time_ms=(end_time - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=0,
            success=False,
            error=str(e)
        )


# =============================================================================
# RAG ANSWER - Single Model (backward compatible)
# =============================================================================
def rag_answer(combined_records, user_prompt, model_key="gemma", max_tokens=500):
    """Get RAG answer from a single model using LangChain"""
    # Sanitize user prompt - remove any problematic characters
    user_prompt = str(user_prompt).strip()
    if not user_prompt:
        return "Error: Empty query provided"
    
    context = build_context(combined_records)
    
    # Truncate context if too long (most models have 4-8k context)
    if len(context) > 10000:
        context = context[:10000] + "\n\n[Context truncated due to length]\n"
    
    try:
        chain = create_rag_chain(model_key, max_tokens)
        response = chain.invoke({"context": context, "question": user_prompt})
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def rag_answer_streaming(combined_records, user_prompt, model_key="gemma", max_tokens=500):
    """
    Get RAG answer with streaming support.
    Yields response chunks as they are generated.
    """
    user_prompt = str(user_prompt).strip()
    if not user_prompt:
        yield "Error: Empty query provided"
        return
    
    context = build_context(combined_records)
    
    if len(context) > 10000:
        context = context[:10000] + "\n\n[Context truncated due to length]\n"
    
    try:
        chain = create_rag_chain(model_key, max_tokens)
        for chunk in chain.stream({"context": context, "question": user_prompt}):
            yield chunk
    except Exception as e:
        yield f"Error: {str(e)}"


# =============================================================================
# RAG COMPARISON - All 3 Models
# =============================================================================
def rag_compare(combined_records, user_prompt, max_tokens=500) -> ComparisonResult:
    """
    Compare RAG answers from all 3 models using LangChain.
    Returns structured comparison with quantitative metrics.
    """
    # Sanitize user prompt
    user_prompt = str(user_prompt).strip()
    
    context = build_context(combined_records)
    prompt = build_prompt(context, user_prompt)
    
    # Truncate prompt if too long
    if len(prompt) > 12000:
        context = context[:10000] + "\n\n[Context truncated due to length]\n"
        prompt = build_prompt(context, user_prompt)
    
    responses = []
    for model_key in MODELS.keys():
        print(f"Calling {MODELS[model_key]['name']}...")
        response = call_model(model_key, prompt, max_tokens)
        responses.append(response)
    
    return ComparisonResult(
        query=user_prompt,
        context_summary=f"{len(combined_records)} records retrieved",
        responses=responses
    )


# =============================================================================
# DISPLAY COMPARISON RESULTS
# =============================================================================
def display_comparison(result: ComparisonResult):
    """Pretty print comparison results"""
    print("\n" + "=" * 80)
    print("LLM COMPARISON RESULTS (LangChain)")
    print("=" * 80)
    print(f"Query: {result.query}")
    print(f"Context: {result.context_summary}")
    print("=" * 80)
    
    for resp in result.responses:
        print(f"\n{'─' * 40}")
        print(f"MODEL: {resp.model_name}")
        print(f"{'─' * 40}")
        
        if resp.success:
            print(f"Response Time: {resp.response_time_ms:.0f}ms")
            print(f"Tokens (est): {resp.input_tokens} input, {resp.output_tokens} output")
            print(f"\nResponse:\n{resp.response}")
        else:
            print(f"ERROR: {resp.error}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("QUANTITATIVE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Time (ms)':<12} {'In Tokens':<12} {'Out Tokens':<12} {'Status':<10}")
    print("-" * 66)
    
    for resp in result.responses:
        status = "✓ OK" if resp.success else "✗ FAIL"
        print(f"{resp.model_name:<20} {resp.response_time_ms:<12.0f} {resp.input_tokens:<12} {resp.output_tokens:<12} {status:<10}")


# =============================================================================
# QUALITATIVE EVALUATION TEMPLATE
# =============================================================================
def create_evaluation_form(result: ComparisonResult) -> str:
    """Generate a qualitative evaluation form for human review"""
    form = """
================================================================================
QUALITATIVE EVALUATION FORM (LangChain)
================================================================================

Query: {query}

Instructions: Rate each model's response on a scale of 1-5 for each criterion.
(1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent)

""".format(query=result.query)

    for i, resp in enumerate(result.responses, 1):
        form += f"""
--------------------------------------------------------------------------------
MODEL {i}: {resp.model_name}
--------------------------------------------------------------------------------
Response: 
{resp.response[:500]}{'...' if len(resp.response) > 500 else ''}

CRITERIA                                          SCORE (1-5)    NOTES
────────────────────────────────────────────────────────────────────────────────
1. Relevance (answers the question)               [ ]            ____________
2. Accuracy (correct based on KG data)            [ ]            ____________
3. Completeness (covers all relevant info)        [ ]            ____________
4. Conciseness (no unnecessary info)              [ ]            ____________
5. Naturalness (fluent, readable)                 [ ]            ____________
6. Groundedness (no hallucinations)               [ ]            ____________

TOTAL SCORE: ___/30
"""

    form += """
================================================================================
OVERALL COMPARISON
================================================================================
Best Model for this Query: _______________
Reasoning: ________________________________________________________________
__________________________________________________________________________

Evaluator: _______________    Date: _______________
================================================================================
"""
    return form


# =============================================================================
# TEST SUITE FOR EVALUATION
# =============================================================================
TEST_QUERIES = [
    "What electronics products are available in Ibiapine?",
    "Show me products with high review scores",
    "Which products had delivery delays?",
    "What is the average price of electronics?",
    "Find products with positive reviews in São Paulo",
]


def run_evaluation_suite(get_records_func):
    """
    Run evaluation suite on all test queries.
    
    Args:
        get_records_func: Function that takes a query and returns combined records
    
    Returns:
        List of ComparisonResult objects
    """
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {i}/{len(TEST_QUERIES)}] {query}")
        records = get_records_func(query)
        result = rag_compare(records, query)
        results.append(result)
        display_comparison(result)
    
    return results


# =============================================================================
# LANGCHAIN-SPECIFIC UTILITIES
# =============================================================================
def get_available_models() -> Dict[str, Dict[str, str]]:
    """Return available models configuration"""
    return MODELS.copy()


def clear_llm_cache():
    """Clear the LLM cache to free memory"""
    global _llm_cache
    _llm_cache = {}


def create_custom_chain(
    model_key: str,
    prompt_template: str,
    input_variables: List[str],
    max_tokens: int = 500
):
    """
    Create a custom LangChain chain with a user-defined prompt template.
    
    Args:
        model_key: Key of the model to use
        prompt_template: Custom prompt template string
        input_variables: List of variable names in the template
        max_tokens: Maximum tokens for response
    
    Returns:
        LangChain chain ready to be invoked
    """
    llm = get_cached_llm(model_key, max_tokens)
    prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)
    
    def invoke_llm(prompt_value) -> str:
        # Convert StringPromptValue to string
        prompt_str = prompt_value.to_string() if hasattr(prompt_value, 'to_string') else str(prompt_value)
        return llm.invoke(prompt_str)
    
    chain = prompt | RunnableLambda(invoke_llm)
    return chain


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("LangChain LLM Module - Example Usage")
    print("=" * 50)
    
    # Test with sample records
    sample_records = [
        {
            "product_id": "abc123",
            "product_category_name": "electronics",
            "price": 299.99,
            "review_score": 4.5,
            "customer_city": "São Paulo"
        },
        {
            "product_id": "def456",
            "product_category_name": "electronics",
            "price": 199.99,
            "review_score": 4.0,
            "customer_city": "Rio de Janeiro"
        }
    ]
    
    question = "What electronics products are available?"
    
    print(f"\nQuestion: {question}")
    print(f"Records: {len(sample_records)}")
    print("\nGetting answer from Gemma model...")
    
    answer = rag_answer(sample_records, question, model_key="gemma")
    print(f"\nAnswer:\n{answer}")
