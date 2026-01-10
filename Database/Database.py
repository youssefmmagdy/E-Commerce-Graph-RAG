from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
def close():
    """Close database connection"""
    driver.close()

def execute_query(query, params):
    """Execute a query and return results"""
    with driver.session() as session:
        result = session.run(query, params)
        return [record for record in result]

def update_node(node_label, identifier, identifier_value, updates):
    """Update node properties"""
    set_clause = ', '.join([f"n.{key} = ${key}" for key in updates.keys()])
    query = f"""
        MATCH (n:{node_label} {{{identifier}: ${identifier}}})
        SET {set_clause}
    """
    params = {'identifier_value': identifier_value}
    params.update(updates)

    with driver.session() as session:
        session.run(query, **params)
