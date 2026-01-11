import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re
import os


nlp = spacy.load("en_core_web_sm")

# Get the directory of this file to build proper path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'Ecommerce_KG_Optimized_translated.csv')

df = pd.read_csv(csv_path)

# =============================================================================
# COLUMN NAMES AS ENTITIES
# =============================================================================
# All valid column names from the CSV - these are recognized as entity types
CSV_COLUMNS = [
    'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp',
    'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date', 'customer_unique_id', 'customer_city', 'customer_state',
    'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price',
    'freight_value', 'product_category_name', 'product_description_lenght', 'product_photos_qty',
    'review_id', 'review_score', 'review_comment_title', 'review_comment_message',
    'review_creation_date', 'delivery_delay_days', 'review_length', 'sentiment_group'
]

# Create lookup for column name variations (with underscore, with space, without separator)
COLUMN_PATTERNS = {}
for col in CSV_COLUMNS:
    # Original with underscore
    COLUMN_PATTERNS[col.lower()] = col
    # With space instead of underscore
    COLUMN_PATTERNS[col.replace('_', ' ').lower()] = col
    # Without any separator (e.g., "orderid" matches "order_id")
    COLUMN_PATTERNS[col.replace('_', '').lower()] = col

# Extract unique values for matching
product_category_name = df['product_category_name'].dropna().unique().tolist()
customer_cities = df['customer_city'].dropna().unique().tolist()
customer_states = df['customer_state'].dropna().unique().tolist()  # These are abbreviations like SP, RJ, MG

# Common English words to exclude from city matching
COMMON_WORDS = {
    'that', 'this', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how',
    'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did', 'get',
    'let', 'put', 'say', 'she', 'too', 'use', 'from', 'have', 'been', 'more',
    'when', 'will', 'with', 'they', 'been', 'call', 'come', 'each', 'find',
    'first', 'than', 'then', 'what', 'where', 'which', 'about', 'after',
    'also', 'back', 'made', 'make', 'many', 'most', 'must', 'only', 'over',
    'such', 'take', 'into', 'just', 'know', 'your', 'could', 'order', 'orders',
    'customer', 'customers', 'product', 'products', 'seller', 'sellers',
    'review', 'reviews', 'state', 'states', 'city', 'cities', 'price', 'delay',
    'show', 'get', 'list', 'find', 'search', 'above', 'below', 'rating', 'score'
}

# Filter out cities that are common words (case-insensitive)
customer_cities_filtered = [c for c in customer_cities if c.lower() not in COMMON_WORDS and len(c) > 2]

# Get actual IDs from database for validation
all_product_ids = set(df['product_id'].dropna().unique().tolist()) if 'product_id' in df.columns else set()
all_customer_ids = set(df['customer_id'].dropna().unique().tolist()) if 'customer_id' in df.columns else set()
all_seller_ids = set(df['seller_id'].dropna().unique().tolist()) if 'seller_id' in df.columns else set()
all_order_ids = set(df['order_id'].dropna().unique().tolist()) if 'order_id' in df.columns else set()

# Initialize PhraseMatchers
matcher_category = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher_state = PhraseMatcher(nlp.vocab)
matcher_city = PhraseMatcher(nlp.vocab, attr="LOWER")

category_patterns = [nlp.make_doc(category) for category in product_category_name]
state_patterns = [nlp.make_doc(state) for state in customer_states]
city_patterns = [nlp.make_doc(city.lower()) for city in customer_cities_filtered]

matcher_category.add("PRODUCT_CATEGORY", category_patterns)
matcher_state.add("STATE", state_patterns)
matcher_city.add("CITY", city_patterns)

# =============================================================================
# REGEX PATTERNS FOR QUERY PARAMETERS
# =============================================================================
# IDs in this dataset are 32-character hexadecimal strings
# Example: e481f51cbdc54678b7cc49136f2d6af7
HEX_ID_PATTERN = re.compile(r'\b([a-f0-9]{32})\b', re.IGNORECASE)

# Pattern for explicit ID mentions (with keyword before the ID)
PRODUCT_ID_PATTERN = re.compile(r'product[_\s]?(?:id)?[:\s]+([a-f0-9]{32})', re.IGNORECASE)
CUSTOMER_ID_PATTERN = re.compile(r'customer[_\s]?(?:id)?[:\s]+([a-f0-9]{32})', re.IGNORECASE)
SELLER_ID_PATTERN = re.compile(r'seller[_\s]?(?:id)?[:\s]+([a-f0-9]{32})', re.IGNORECASE)
ORDER_ID_PATTERN = re.compile(r'order[_\s]?(?:id)?[:\s]+([a-f0-9]{32})', re.IGNORECASE)

# Alternative pattern: "ID abc123" or just the ID if it exists in database
GENERIC_ID_PATTERN = re.compile(r'(?:id[:\s]+)?([a-f0-9]{32})', re.IGNORECASE)

# Pattern for numeric values with context
# Matches: "review_score above 4", "rating above 4", "score >= 3.5"
RATING_PATTERN = re.compile(r'(?:review[_\s]?score|rating|score|rated)[:\s]*(?:above|over|>=?|at least)?\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
RATING_PATTERN_2 = re.compile(r'(?:above|over|>=?|at least)\s*(\d+(?:\.\d+)?)\s*(?:review[_\s]?score|rating|score|stars?)?', re.IGNORECASE)

# Matches: "delivery_delay_days over 5", "delay over 5 days", "5 days delay"
DELAY_PATTERN = re.compile(r'(?:delivery[_\s]?delay[_\s]?days?|delay(?:ed)?)[:\s]*(?:over|above|>=?|more than)?\s*(\d+)\s*(?:days?)?', re.IGNORECASE)
DELAY_PATTERN_2 = re.compile(r'(\d+)\s*(?:days?)\s*(?:delivery[_\s]?delay[_\s]?days?|delay(?:ed)?)', re.IGNORECASE)
DELAY_PATTERN_3 = re.compile(r'(?:over|above|more than)\s*(\d+)\s*(?:days?)', re.IGNORECASE)

MIN_ORDERS_PATTERN = re.compile(r'(\d+)\+?\s*orders?', re.IGNORECASE)
MIN_ORDERS_PATTERN_2 = re.compile(r'(?:at least|minimum|min|>=?)\s*(\d+)\s*orders?', re.IGNORECASE)

# State abbreviations (Brazilian states)
STATE_ABBREV_MAP = {
    'SP': 'São Paulo', 'RJ': 'Rio de Janeiro', 'MG': 'Minas Gerais',
    'RS': 'Rio Grande do Sul', 'PR': 'Paraná', 'SC': 'Santa Catarina',
    'BA': 'Bahia', 'PE': 'Pernambuco', 'CE': 'Ceará', 'PA': 'Pará',
    'MA': 'Maranhão', 'GO': 'Goiás', 'AM': 'Amazonas', 'ES': 'Espírito Santo',
    'PB': 'Paraíba', 'RN': 'Rio Grande do Norte', 'MT': 'Mato Grosso',
    'AL': 'Alagoas', 'PI': 'Piauí', 'MS': 'Mato Grosso do Sul', 'DF': 'Distrito Federal',
    'SE': 'Sergipe', 'RO': 'Rondônia', 'TO': 'Tocantins', 'AC': 'Acre',
    'AP': 'Amapá', 'RR': 'Roraima'
}


def extract_column_entities(text):
    """
    Extract column names and their values from text.
    Recognizes column names with underscores, spaces, or no separator.
    
    Returns: List of (column_name, value, label) tuples
    """
    entities = []
    text_lower = text.lower()
    
    # Build regex patterns for each column variation
    for col in CSV_COLUMNS:
        # Create patterns for different variations
        variations = [
            col,                          # order_id
            col.replace('_', ' '),        # order id
            col.replace('_', '')          # orderid
        ]
        
        for var in variations:
            # Pattern 1: column_name followed by value (with optional separator)
            # Matches: "order_id abc123", "order_id: abc123", "order id abc123"
            pattern1 = re.compile(
                r'\b' + re.escape(var) + r'[:\s]+([^\s,]+)',
                re.IGNORECASE
            )
            
            # Pattern 2: column_name with comparison operators and numeric value
            # Matches: "review_score above 4", "delivery_delay_days over 5"
            pattern2 = re.compile(
                r'\b' + re.escape(var) + r'[:\s]*(?:above|over|>=?|>|at least|more than)\s*(\d+(?:\.\d+)?)',
                re.IGNORECASE
            )
            
            # Try pattern2 first (for numeric comparisons)
            match = pattern2.search(text)
            if match:
                value = match.group(1).strip()
                entities.append((col, value, col.upper()))
                break
            
            # Try pattern1 for general values
            match = pattern1.search(text)
            if match:
                value = match.group(1).strip()
                # Clean up value (remove trailing punctuation)
                value = re.sub(r'[.,;:!?]+$', '', value)
                if value and len(value) > 0:
                    entities.append((col, value, col.upper()))
                    break
    
    return entities


def find_id_by_type(text, id_type):
    """
    Find an ID in text, checking if it exists in the database.
    id_type: 'product', 'customer', 'seller', 'order'
    """
    # First try explicit pattern
    if id_type == 'product':
        match = PRODUCT_ID_PATTERN.search(text)
        id_set = all_product_ids
    elif id_type == 'customer':
        match = CUSTOMER_ID_PATTERN.search(text)
        id_set = all_customer_ids
    elif id_type == 'seller':
        match = SELLER_ID_PATTERN.search(text)
        id_set = all_seller_ids
    elif id_type == 'order':
        match = ORDER_ID_PATTERN.search(text)
        id_set = all_order_ids
    else:
        return None
    
    if match:
        return match.group(1)
    
    # Try to find any 32-char hex ID and check if it exists in the database
    all_ids = HEX_ID_PATTERN.findall(text)
    for found_id in all_ids:
        if found_id.lower() in {x.lower() for x in id_set}:
            return found_id
    
    return None


def extract_query_parameters(text):
    """
    Extract parameters needed for the 10 Cypher queries.
    
    Returns: dict with keys matching query parameters:
        - category: Product category name (Query 1, 2)
        - city: Customer city (Query 2)
        - state: Customer state abbreviation like SP, RJ (Query 7)
        - min_rating: Minimum rating threshold (Query 3)
        - product_id: Product ID - 32 char hex (Query 4)
        - customer_id: Customer ID - 32 char hex (Query 5)
        - seller_id: Seller ID - 32 char hex (Query 8)
        - order_id: Order ID - 32 char hex
        - delay_days: Delivery delay threshold (Query 6)
        - min_orders: Minimum orders for repeat customers (Query 10)
    """
    params = {}
    
    doc_lower = nlp(text.lower())
    doc = nlp(text)
    
    # 0. Extract column-based entities first (handles all CSV columns)
    column_entities = extract_column_entities(text)
    for col_name, value, label in column_entities:
        # Map column names to parameter names used in queries
        if col_name == 'review_score':
            try:
                params['min_rating'] = float(value)
            except ValueError:
                pass
        elif col_name == 'delivery_delay_days':
            try:
                params['delay_days'] = int(value)
            except ValueError:
                pass
        elif col_name == 'product_id':
            params['product_id'] = value
        elif col_name == 'customer_id':
            params['customer_id'] = value
        elif col_name == 'seller_id':
            params['seller_id'] = value
        elif col_name == 'order_id':
            params['order_id'] = value
        elif col_name == 'customer_city':
            params['city'] = value
        elif col_name == 'customer_state':
            params['state'] = value.upper()
        elif col_name == 'product_category_name':
            params['category'] = value
    
    # 1. Extract CATEGORY (Query 1, 2) - if not already found
    if 'category' not in params:
        matches = matcher_category(doc_lower)
        for match_id, start, end in matches:
            params['category'] = doc_lower[start:end].text.strip()
            break  # Take first match
    
    # 2. Extract CITY (Query 2, 10) - if not already found
    if 'city' not in params:
        matches = matcher_city(doc_lower)
        for match_id, start, end in matches:
            # Get original case from customer_cities list
            matched_text = doc_lower[start:end].text.strip().lower()
            # Skip common English words
            if matched_text in COMMON_WORDS:
                continue
            for city in customer_cities_filtered:
                if city.lower() == matched_text:
                    params['city'] = city
                    break
            if 'city' in params:
                break
    
    # 3. Extract STATE (Query 7) - handles both abbreviations (SP) and full names
    if 'state' not in params:
        # First check for abbreviations in text
        for abbrev in customer_states:  # customer_states contains abbreviations like SP, RJ
            pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                params['state'] = abbrev.upper()
                break
    
    # Also check for full state names
    if 'state' not in params:
        for abbrev, full_name in STATE_ABBREV_MAP.items():
            if full_name.lower() in text.lower():
                params['state'] = abbrev
                break
    
    # 4. Extract RATING (Query 3) - if not already found
    if 'min_rating' not in params:
        rating_match = RATING_PATTERN.search(text) or RATING_PATTERN_2.search(text)
        if rating_match:
            params['min_rating'] = float(rating_match.group(1))
    
    # 5. Extract PRODUCT_ID (Query 4) - if not already found
    if 'product_id' not in params:
        product_id = find_id_by_type(text, 'product')
        if product_id:
            params['product_id'] = product_id
    
    # 6. Extract CUSTOMER_ID (Query 5) - if not already found
    if 'customer_id' not in params:
        customer_id = find_id_by_type(text, 'customer')
        if customer_id:
            params['customer_id'] = customer_id
    
    # 7. Extract DELAY_DAYS (Query 6) - if not already found
    if 'delay_days' not in params:
        delay_match = DELAY_PATTERN.search(text) or DELAY_PATTERN_2.search(text) or DELAY_PATTERN_3.search(text)
        if delay_match:
            params['delay_days'] = int(delay_match.group(1))
    
    # 8. Extract SELLER_ID (Query 8)
    seller_id = find_id_by_type(text, 'seller')
    if seller_id:
        params['seller_id'] = seller_id
    
    # 9. Extract MIN_ORDERS (Query 10)
    orders_match = MIN_ORDERS_PATTERN.search(text) or MIN_ORDERS_PATTERN_2.search(text)
    if orders_match:
        params['min_orders'] = int(orders_match.group(1))
    
    # 10. Extract ORDER_ID
    order_id = find_id_by_type(text, 'order')
    if order_id:
        params['order_id'] = order_id
    
    return params


def detect_query_type(text, params=None):
    """
    Detect which of the 10 queries best matches the user's intent.
    
    Returns: (query_number, confidence, description)
    """
    if params is None:
        params = extract_query_parameters(text)
    
    text_lower = text.lower()
    
    # Query detection rules
    detections = []
    
    # Query 1: Products by category (just category, no city)
    if 'category' in params and 'product' in text_lower:
        detections.append((1, 0.8, "Products by category"))
    
    # Query 2: Products by category AND city
    elif 'category' in params and 'city' in params:
        detections.append((2, 0.95, "Products by category and city"))
    
    # Query 3: Products by city (city but no category, and mentions product)
    elif 'city' in params and 'product' in text_lower:
        detections.append((3, 0.9, "Products by city"))
    
    # Query 4: Reviews for product
    elif 'product_id' in params or ('review' in text_lower and 'product' in text_lower):
        detections.append((4, 0.9, "Reviews for product"))
    
    # Query 5: Orders by customer
    elif 'customer_id' in params or ('order' in text_lower and 'customer' in text_lower):
        detections.append((5, 0.85, "Orders by customer"))
    
    # Query 6: Orders with delays
    elif 'delay_days' in params or 'delayed days' in params or 'delay' in text_lower or 'delayed' in text_lower or 'delivery_delay_days' in text_lower:
        detections.append((6, 0.9, "Orders with delivery delays"))
    
    # Query 7: Customers by state
    elif 'state' in params and 'customer' in text_lower:
        detections.append((7, 0.85, "Customers by state"))
    
    # Query 8: Get specific order
    elif 'order_id' in params and 'order' in text_lower:
        detections.append((8, 0.95, "Get specific order"))
    
    # Query 9: Customers that bought from a specific seller
    elif 'seller_id' in params or 'seller' in text_lower and 'customer' in text_lower:
        detections.append((9, 0.85, "Customers by seller"))
    
    # Query 10: Customers by city
    elif 'city' in params and 'customer' in text_lower:
        detections.append((10, 0.9, "Customers by city"))
    
    # Return best match or default
    if detections:
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections[0]
    
    # Default: try to infer from available params
    if 'category' in params:
        return (1, 0.5, "Products by category (inferred)")
    
    return (9, 0.3, "Most popular products (default)")


def extract_entities(text):
    """
    Extract both standard NER entities and custom entities for all 10 queries.
    Recognizes all CSV column names as valid entity types.
    
    Returns: List of (entity_text, entity_label) tuples
    - Any CSV column name (e.g., ORDER_ID, CUSTOMER_ID, REVIEW_SCORE, DELIVERY_DELAY_DAYS)
    - PRODUCT_CATEGORY: Product categories (Query 1, 2)
    - CITY: Customer cities (Query 2, 10)
    - STATE: Customer state abbreviations like SP, RJ (Query 7)
    - GPE: Geographic locations (spaCy NER)
    - PERSON: People (spaCy NER)
    - ORG: Organizations (spaCy NER)
    """
    doc_lower = nlp(text.lower())
    doc = nlp(text)
    entities = []
    
    # 0. Extract column-based entities (e.g., "review_score above 4" -> ("4", "REVIEW_SCORE"))
    column_entities = extract_column_entities(text)
    for col_name, value, label in column_entities:
        entities.append((value, label))
    
    # 1. Custom product categories (Query 1, 2)
    matches = matcher_category(doc_lower)
    for match_id, start, end in matches:
        category_text = doc_lower[start:end].text.strip()
        entities.append((category_text, "PRODUCT_CATEGORY"))
    
    # 2. Custom City (Query 2, 10)
    matches = matcher_city(doc_lower)
    for match_id, start, end in matches:
        city_text = doc_lower[start:end].text.strip()
        # Skip common English words
        if city_text.lower() in COMMON_WORDS:
            continue
        # Get original case
        for city in customer_cities_filtered:
            if city.lower() == city_text.lower():
                entities.append((city, "CITY"))
                break
    
    # 3. State abbreviations (Query 7)
    for abbrev in customer_states:
        pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            entities.append((abbrev.upper(), "STATE"))
            break
    
    # Also check full state names
    for abbrev, full_name in STATE_ABBREV_MAP.items():
        if full_name.lower() in text.lower():
            entities.append((abbrev, "STATE"))
            break
    
    # 4. Extract IDs using new find_id_by_type (Query 4, 5, 8)
    product_id = find_id_by_type(text, 'product')
    if product_id:
        entities.append((product_id, "PRODUCT_ID"))
    
    customer_id = find_id_by_type(text, 'customer')
    if customer_id:
        entities.append((customer_id, "CUSTOMER_ID"))
    
    seller_id = find_id_by_type(text, 'seller')
    if seller_id:
        entities.append((seller_id, "SELLER_ID"))
    
    order_id = find_id_by_type(text, 'order')
    if order_id:
        entities.append((order_id, "ORDER_ID"))
    
    # 5. Extract numeric parameters (Query 3, 6, 10)
    rating_match = RATING_PATTERN.search(text) or RATING_PATTERN_2.search(text)
    if rating_match:
        entities.append((rating_match.group(1), "RATING"))
    
    delay_match = DELAY_PATTERN.search(text) or DELAY_PATTERN_2.search(text) or DELAY_PATTERN_3.search(text)
    if delay_match:
        entities.append((delay_match.group(1), "DELAY_DAYS"))
    
    orders_match = MIN_ORDERS_PATTERN.search(text) or MIN_ORDERS_PATTERN_2.search(text)
    if orders_match:
        entities.append((orders_match.group(1), "MIN_ORDERS"))
    
    # 6. Standard NER entities
    for ent in doc.ents:
        if ent.label_ in nlp.pipe_labels['ner']:
            if ent.text.strip() not in [e[0] for e in entities]:
                entities.append((ent.text.strip(), ent.label_))

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for ent_text, ent_label in entities:
        key = (ent_text.lower(), ent_label)
        if key not in seen:
            seen.add(key)
            unique_entities.append((ent_text, ent_label))
    
    return unique_entities


def get_full_extraction(text):
    """
    Get complete extraction results for a query.
    
    Returns: dict with:
        - entities: List of (text, label) tuples
        - params: Dict of query parameters
        - query_type: (number, confidence, description)
    """
    entities = extract_entities(text)
    params = extract_query_parameters(text)
    query_type = detect_query_type(text, params)
    
    return {
        'entities': entities,
        'params': params,
        'query_type': query_type
    }
