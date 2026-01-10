"""
Map extracted entities to Cypher query parameters
"""

def entities_to_query_params(entities, limit=20):
    """
    Convert extracted entities to query parameters.
    Handles all CSV column names as entity labels.
    
    Args:
        entities: List of (entity_text, entity_label) tuples
        limit: Result limit (default: 20)
    
    Returns:
        Dictionary of query parameters
    """
    # Initialize all possible parameters with default values
    params = {
        "limit": limit,
        # Core query parameters (used in Cypher queries)
        "category": None,
        "city": None,
        "state": None,
        "customer_id": None,
        "product_id": None,
        "seller_id": None,
        "order_id": None,
        "min_rating": None,
        "delay_days": None,
        # All CSV columns as potential parameters
        "order_status": None,
        "order_purchase_timestamp": None,
        "order_approved_at": None,
        "order_delivered_carrier_date": None,
        "order_delivered_customer_date": None,
        "order_estimated_delivery_date": None,
        "customer_unique_id": None,
        "order_item_id": None,
        "shipping_limit_date": None,
        "price": None,
        "freight_value": None,
        "product_description_lenght": None,
        "product_photos_qty": None,
        "review_id": None,
        "review_score": None,
        "review_comment_title": None,
        "review_comment_message": None,
        "review_creation_date": None,
        "delivery_delay_days": None,
        "review_length": None,
        "sentiment_group": None,
    }
    
    for entity_text, entity_label in entities:
        label_upper = entity_label.upper()
        label_lower = entity_label.lower()
        
        # Handle standard entity labels
        if label_upper == 'PRODUCT_CATEGORY':
            params["category"] = entity_text.lower()
        elif label_upper in ['GPE', 'CITY', 'CUSTOMER_CITY']:
            params["city"] = entity_text
        elif label_upper in ['STATE', 'CUSTOMER_STATE']:
            params["state"] = entity_text.upper()
        elif label_upper == 'CUSTOMER_ID':
            params["customer_id"] = entity_text
        elif label_upper == 'PRODUCT_ID':
            params["product_id"] = entity_text
        elif label_upper == 'SELLER_ID':
            params["seller_id"] = entity_text
        elif label_upper == 'ORDER_ID':
            params["order_id"] = entity_text
        elif label_upper in ['RATING', 'REVIEW_SCORE']:
            try:
                params["min_rating"] = float(entity_text)
                params["review_score"] = float(entity_text)
            except ValueError:
                pass
        elif label_upper in ['DELAY_DAYS', 'DELIVERY_DELAY_DAYS']:
            try:
                params["delay_days"] = int(entity_text)
                params["delivery_delay_days"] = int(entity_text)
            except ValueError:
                pass
        elif label_upper == 'PRICE':
            try:
                params["price"] = float(entity_text)
            except ValueError:
                pass
        elif label_upper == 'FREIGHT_VALUE':
            try:
                params["freight_value"] = float(entity_text)
            except ValueError:
                pass
        elif label_upper == 'PRODUCT_CATEGORY_NAME':
            params["category"] = entity_text.lower()
        # Handle all other CSV column labels directly
        elif label_lower in params:
            params[label_lower] = entity_text
    
    return params


# # Example usage
# if __name__ == "__main__":
#     entities = [('utilidades_domesticas', 'PRODUCT_CATEGORY'), ('rio de janeiro', 'GPE')]
#     params = entities_to_query_params(entities)
#     print("Converted parameters:")
#     print(params)
#     # Output: {'limit': 20, 'category': 'utilidades_domesticas', 'city': 'rio de janeiro', ...}
