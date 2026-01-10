import sys
import os
# Add parent directory to path so imports work when running from Baseline folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from Baseline.EntityExtractor import extract_entities, extract_query_parameters, detect_query_type
from Database.Database import close, update_node, execute_query
from Queries.Queries import get_cypher_query_by_number
from Baseline.EntityMapper import entities_to_query_params

def get_baseline_records(user_prompt, query_number=None):
    """
    Get records from Neo4j using baseline Cypher queries.
    
    Args:
        user_prompt: Natural language query from user
        query_number: Optional - force a specific query (1-10). 
                     If None, auto-detects based on extracted entities.
    """
    entities = extract_entities(user_prompt)
    print("📍 Extracted Entities:", entities)
    
    # Get query parameters
    params = entities_to_query_params(entities)
    print("📍 Query Parameters:", params)
    
    # Auto-detect query type if not specified
    if query_number is None:
        query_type_info = detect_query_type(user_prompt, extract_query_parameters(user_prompt))
        query_number = query_type_info[0]
        print(f"📍 Auto-detected Query #{query_number}: {query_type_info[2]} (confidence: {query_type_info[1]:.0%})")
    
    # Execute the query
    try:
        records = execute_query(get_cypher_query_by_number(query_number), params)
    except Exception as e:
        print(f"⚠️ Query execution error: {e}")
        return []
    
    # Convert Neo4j Records to dict format - return all columns
    results = []
    for record in records:
        result_dict = {"source": "baseline", "query_type": query_number}
        for key in record.keys():
            result_dict[key] = record[key]
        results.append(result_dict)
    
    print(f"\n✓ Retrieved {len(results)} baseline records.")
    return results

