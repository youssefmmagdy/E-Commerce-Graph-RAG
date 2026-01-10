# =========================
# BASELINE CYPHER QUERIES
# Milestone 3 – Graph-RAG
# All queries return flattened columns (no Node objects)
# Each query returns total_count (without limit) + limited records
# =========================

# 1. Products by category
QUERY_PRODUCTS_BY_CATEGORY = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE p.product_category_name = $category
    RETURN count(DISTINCT p.product_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE p.product_category_name = $category
RETURN DISTINCT 
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty AS product_photos_qty,
    oi.price AS price,
    oi.freight_value AS freight_value,
    total_count
LIMIT 10
"""

# 2. Products by category and city
QUERY_PRODUCTS_BY_CATEGORY_AND_CITY = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE toLower(c.customer_city) = toLower($city)
      AND p.product_category_name = $category
    RETURN count(DISTINCT o.order_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE toLower(c.customer_city) = toLower($city)
  AND p.product_category_name = $category
RETURN DISTINCT 
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    o.order_approved_at AS order_approved_at,
    o.order_delivered_carrier_date AS order_delivered_carrier_date,
    o.order_delivered_customer_date AS order_delivered_customer_date,
    o.order_estimated_delivery_date AS order_estimated_delivery_date,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty AS product_photos_qty,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
LIMIT 10
"""

# 3. Products in a specifc city
QUERY_PRODUCTS_BY_CITY = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE c.customer_city = $city
    RETURN count(DISTINCT p.product_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE toLower(c.customer_city) = toLower($city)
RETURN DISTINCT 
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty AS product_photos_qty,
    oi.price AS price,
    oi.freight_value AS freight_value,
    c.customer_city,
    total_count
LIMIT 10
"""

# 4. Reviews for a specific product
QUERY_REVIEWS_FOR_PRODUCT = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product),
          (r:Review)-[:REVIEWS]->(o)
    WHERE p.product_id = $product_id
    RETURN count(DISTINCT r.review_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product),
      (r:Review)-[:REVIEWS]->(o)
WHERE p.product_id = $product_id
RETURN 
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    o.order_approved_at AS order_approved_at,
    o.order_delivered_carrier_date AS order_delivered_carrier_date,
    o.order_delivered_customer_date AS order_delivered_customer_date,
    o.order_estimated_delivery_date AS order_estimated_delivery_date,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty AS product_photos_qty,
    oi.price AS price,
    oi.freight_value AS freight_value,
    r.review_id AS review_id,
    r.review_score AS review_score,
    r.review_comment_title AS review_comment_title,
    r.review_comment_message AS review_comment_message,
    r.review_creation_date AS review_creation_date,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
ORDER BY r.review_creation_date DESC
LIMIT 10
"""

# 5. Orders by customer
QUERY_ORDERS_BY_CUSTOMER = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE c.customer_id = $customer_id
    RETURN count(DISTINCT o.order_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE c.customer_id = $customer_id
RETURN 
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    o.order_approved_at AS order_approved_at,
    o.order_delivered_carrier_date AS order_delivered_carrier_date,
    o.order_delivered_customer_date AS order_delivered_customer_date,
    o.order_estimated_delivery_date AS order_estimated_delivery_date,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
ORDER BY o.order_purchase_timestamp DESC
LIMIT 10
"""

# 6. Orders with delivery delays
QUERY_ORDERS_WITH_DELAYS = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE o.delivery_delay_days > $delay_days
    RETURN count(DISTINCT o.order_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE o.delivery_delay_days > $delay_days
RETURN 
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    o.order_approved_at AS order_approved_at,
    o.order_delivered_carrier_date AS order_delivered_carrier_date,
    o.order_delivered_customer_date AS order_delivered_customer_date,
    o.order_estimated_delivery_date AS order_estimated_delivery_date,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
ORDER BY o.delivery_delay_days DESC
LIMIT 10
"""

# 7. Customers by state
QUERY_CUSTOMERS_BY_STATE = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE c.customer_state = $state
    RETURN count(DISTINCT o.order_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE c.customer_state = $state
RETURN DISTINCT
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
LIMIT 10
"""

# 8. Get a specific order
QUERY_GET_SPECIFIC_ORDER = """
MATCH (o:Order {order_id: $order_id})<-[:PLACED]-(c:Customer),
      (o)-[:CONTAINS]->(oi:Order_Item)-[:REFERS_TO]->(p:Product)
RETURN 
    o.order_id AS order_id,
    c.customer_id AS customer_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    o.order_approved_at AS order_approved_at,
    o.order_delivered_carrier_date AS order_delivered_carrier_date,
    o.order_delivered_customer_date AS order_delivered_customer_date,
    o.order_estimated_delivery_date AS order_estimated_delivery_date,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    1 AS total_count
LIMIT 10
"""

# 9. Customers that bought from a specific seller
QUERY_CUSTOMERS_BY_SELLER = """
CALL {
    MATCH (s:Seller)<-[:SOLD_BY]-(oi:Order_Item)
          <-[:CONTAINS]-(o:Order)<-[:PLACED]-(c:Customer),
          (oi)-[:REFERS_TO]->(p:Product)
    WHERE s.seller_id = $seller_id
    RETURN count(DISTINCT c.customer_id) AS total_count
}
MATCH (s:Seller)<-[:SOLD_BY]-(oi:Order_Item)
      <-[:CONTAINS]-(o:Order)<-[:PLACED]-(c:Customer),
      (oi)-[:REFERS_TO]->(p:Product)
WHERE s.seller_id = $seller_id
RETURN DISTINCT
    c.customer_id AS customer_id,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    o.order_id AS order_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    s.seller_id AS seller_id,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
LIMIT 10
"""

# 10. Customers that ordered in a specific city
QUERY_CUSTOMERS_BY_CITY = """
CALL {
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(oi:Order_Item)
          -[:REFERS_TO]->(p:Product)
    WHERE toLower(c.customer_city) = toLower($city)
    RETURN count(DISTINCT c.customer_id) AS total_count
}
MATCH (c:Customer)-[:PLACED]->(o:Order)
      -[:CONTAINS]->(oi:Order_Item)
      -[:REFERS_TO]->(p:Product)
WHERE toLower(c.customer_city) = toLower($city)
RETURN DISTINCT
    c.customer_id AS customer_id,
    c.customer_city AS customer_city,
    c.customer_state AS customer_state,
    o.order_id AS order_id,
    o.order_status AS order_status,
    o.order_purchase_timestamp AS order_purchase_timestamp,
    p.product_id AS product_id,
    p.product_category_name AS product_category_name,
    oi.price AS price,
    oi.freight_value AS freight_value,
    o.delivery_delay_days AS delivery_delay_days,
    total_count
LIMIT 10
"""

def get_cypher_query_by_number(n):
    """
    Returns the nth Cypher query template (1-based index).

    Args:
        n (int): Query number (1-10)

    Returns:
        str: Cypher query string

    Raises:
        IndexError: If n is not in the valid range
    """
    queries = [
        QUERY_PRODUCTS_BY_CATEGORY,              # 1
        QUERY_PRODUCTS_BY_CATEGORY_AND_CITY,     # 2
        QUERY_PRODUCTS_BY_CITY,                   # 3
        QUERY_REVIEWS_FOR_PRODUCT,                # 4
        QUERY_ORDERS_BY_CUSTOMER,                 # 5
        QUERY_ORDERS_WITH_DELAYS,                 # 6
        QUERY_CUSTOMERS_BY_STATE,                 # 7
        QUERY_GET_SPECIFIC_ORDER,                 # 8
        QUERY_CUSTOMERS_BY_SELLER,                # 9
        QUERY_CUSTOMERS_BY_CITY                   # 10
    ]

    if not (1 <= n <= len(queries)):
        raise IndexError(f"Query number must be between 1 and {len(queries)}")

    return queries[n - 1]
