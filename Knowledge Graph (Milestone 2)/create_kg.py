import pandas as pd
import spacy
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np
from collections import Counter, defaultdict


df = pd.read_csv('Ecommerce_KG_Optimized.csv')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.head()

nlp = spacy.load("en_core_web_sm")

doc = nlp(df['review_comment_message'][0])
print(doc)
print(doc.ents)

# Entities
states_entity = []
reviews_entity = []
sellers_entity = []
products_entity = []
order_items_entity = []
orders_entity = []
customers_entity = []

# Relationships
customer_placed_order = []
order_contains_orderItem = []
orderItem_refersTo_product = []
orderItem_soldBy_seller = []
customer_wrote_review = []
review_reviews_order = []
customer_locatedIn_state = []

for i, row in df.iterrows():
    # if i > 10000:
    #     break
    # Entities
    customers_entity.append({
        'customer_id': row['customer_id'],
        'customer_unique_id': row['customer_unique_id'],
        'customer_city': row['customer_city'],
        'customer_state': row['customer_state']
    })
    orders_entity.append({
        'order_id': row['order_id'],
        'order_status': row['order_status'],
        'order_purchase_timestamp': row['order_purchase_timestamp'],
        'order_approved_at': row['order_approved_at'],
        'order_delivered_carrier_date': row['order_delivered_carrier_date'],
        'order_delivered_customer_date': row['order_delivered_customer_date'],
        'order_estimated_delivery_date': row['order_estimated_delivery_date'],
        'delivery_delay_days': row['delivery_delay_days']
    })
    order_items_entity.append({
        'order_item_id': row['order_item_id'],
        'price': row['price'],
        'freight_value': row['freight_value']
    })
    products_entity.append({
        'product_id': row['product_id'],
        'product_category_name': row['product_category_name'],
        'product_description_lenght': row['product_description_lenght'],
        'product_photos_qty': row['product_photos_qty']
    })
    sellers_entity.append({
        'seller_id': row['seller_id']
    })
    reviews_entity.append({
        'review_id': row['review_id'],
        'review_score': row['review_score'],
        'review_comment_title': row['review_comment_title'],
        'review_comment_message': row['review_comment_message'],
        'review_creation_date': row['review_creation_date'],
        'review_length': row['review_length']
    })
    states_entity.append({
        'name': row['customer_state']
    })

    # Relationships
    customer_placed_order.append(
        [
            {'customer_id': row['customer_id']}, 
            {'order_id': row['order_id']}
        ]
        )
    order_contains_orderItem.append(
        [
            {'order_id': row['order_id']},
            {'order_item_id': row['order_item_id']}  
        ]
    )
    orderItem_refersTo_product.append(
        [
            {'order_item_id': row['order_item_id']}  ,
            {'product_id': row['product_id']}
        ]
    )
    orderItem_soldBy_seller.append(
        [
            {'order_item_id': row['order_item_id']},
            {'seller_id': row['seller_id']}
        ]
    )
    customer_wrote_review.append(
        [
            {'customer_id': row['customer_id']}, 
            {'review_id': row['review_id']}  
        ]
    )
    review_reviews_order.append(
        [
            {'review_id': row['review_id']},
            {'order_id': row['order_id']}
        ]
    )
    customer_locatedIn_state.append(
        [
            {'customer_id': row['customer_id']},
            {'customer_state': row['customer_state']}  
        ]
    )


def build_kg_nodes(tx):
    """Create all nodes in the knowledge graph"""
    # Clear existing data
    print("Clearing existing data...")
    tx.run("MATCH (n) DETACH DELETE n")
    
    # Create nodes in batches
    batch_size = 5000
    
    # Create Customer nodes
    if customers_entity:
        print(f"Creating {len(customers_entity)} Customer nodes...")
        for i in range(0, len(customers_entity), batch_size):
            batch = customers_entity[i:i+batch_size]
            tx.run(
                """
                UNWIND $customers AS customer
                MERGE (c:Customer {customer_id: customer.customer_id})
                SET c.customer_unique_id = customer.customer_unique_id,
                    c.customer_city = customer.customer_city,
                    c.customer_state = customer.customer_state
                """,
                customers=batch
            )
    
    # Create Order nodes
    if orders_entity:
        print(f"Creating {len(orders_entity)} Order nodes...")
        for i in range(0, len(orders_entity), batch_size):
            batch = orders_entity[i:i+batch_size]
            tx.run(
                """
                UNWIND $orders AS order
                MERGE (o:Order {order_id: order.order_id})
                SET o.order_status = order.order_status,
                    o.order_purchase_timestamp = order.order_purchase_timestamp,
                    o.order_approved_at = order.order_approved_at,
                    o.order_delivered_carrier_date = order.order_delivered_carrier_date,
                    o.order_delivered_customer_date = order.order_delivered_customer_date,
                    o.order_estimated_delivery_date = order.order_estimated_delivery_date,
                    o.delivery_delay_days = order.delivery_delay_days
                """,
                orders=batch
            )
    
    # Create Order_Item nodes
    if order_items_entity:
        print(f"Creating {len(order_items_entity)} Order_Item nodes...")
        for i in range(0, len(order_items_entity), batch_size):
            batch = order_items_entity[i:i+batch_size]
            tx.run(
                """
                UNWIND $order_items AS order_item
                MERGE (oi:Order_Item {order_item_id: order_item.order_item_id})
                SET oi.price = order_item.price,
                    oi.freight_value = order_item.freight_value
                """,
                order_items=batch
            )
    
    # Create Product nodes (deduplicated)
    print(f"Creating Product nodes...")
    unique_products = []
    seen_ids = set()
    for p in products_entity:
        if p['product_id'] not in seen_ids:
            unique_products.append(p)
            seen_ids.add(p['product_id'])
    
    for i in range(0, len(unique_products), batch_size):
        batch = unique_products[i:i+batch_size]
        tx.run(
            """
            UNWIND $products AS product
            MERGE (p:Product {product_id: product.product_id})
            SET p.product_category_name = product.product_category_name,
                p.product_description_lenght = product.product_description_lenght,
                p.product_photos_qty = product.product_photos_qty
            """,
            products=batch
        )
    
    # Create Seller nodes (deduplicated)
    print(f"Creating Seller nodes...")
    unique_sellers = []
    seen_seller_ids = set()
    for s in sellers_entity:
        if s['seller_id'] not in seen_seller_ids:
            unique_sellers.append(s)
            seen_seller_ids.add(s['seller_id'])
    
    for i in range(0, len(unique_sellers), batch_size):
        batch = unique_sellers[i:i+batch_size]
        tx.run(
            """
            UNWIND $sellers AS seller
            MERGE (s:Seller {seller_id: seller.seller_id})
            """,
            sellers=batch
        )
    
    # Create Review nodes
    if reviews_entity:
        print(f"Creating {len(reviews_entity)} Review nodes...")
        for i in range(0, len(reviews_entity), batch_size):
            batch = reviews_entity[i:i+batch_size]
            tx.run(
                """
                UNWIND $reviews AS review
                MERGE (r:Review {review_id: review.review_id})
                SET r.review_score = review.review_score,
                    r.review_comment_title = review.review_comment_title,
                    r.review_comment_message = review.review_comment_message,
                    r.review_creation_date = review.review_creation_date,
                    r.review_length = review.review_length
                """,
                reviews=batch
            )
    
    # Create State nodes (deduplicated)
    print(f"Creating State nodes...")
    unique_states = []
    seen_states = set()
    for s in states_entity:
        if s['name'] not in seen_states:
            unique_states.append(s)
            seen_states.add(s['name'])
    
    tx.run(
        """
        UNWIND $states AS state
        MERGE (st:State {name: state.name})
        """,
        states=unique_states
    )
    
    return "All nodes created successfully"


def build_kg_relationships(tx):
    """Create all relationships in the knowledge graph"""
    batch_size = 5000
    
    # (Customer) –[:PLACED]→ (Order)
    if customer_placed_order:
        print(f"Creating {len(customer_placed_order)} PLACED relationships...")
        for i in range(0, len(customer_placed_order), batch_size):
            batch = customer_placed_order[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (c:Customer {customer_id: rel[0].customer_id})
                MATCH (o:Order {order_id: rel[1].order_id})
                MERGE (c)-[:PLACED]->(o)
                """,
                relationships=batch
            )
    
    # (Order) –[:CONTAINS]→ (OrderItem)
    if order_contains_orderItem:
        print(f"Creating {len(order_contains_orderItem)} CONTAINS relationships...")
        for i in range(0, len(order_contains_orderItem), batch_size):
            batch = order_contains_orderItem[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (o:Order {order_id: rel[0].order_id})
                MATCH (oi:Order_Item {order_item_id: rel[1].order_item_id})
                MERGE (o)-[:CONTAINS]->(oi)
                """,
                relationships=batch
            )
    
    # (OrderItem) -[:REFERS_TO]→(Product)
    if orderItem_refersTo_product:
        print(f"Creating {len(orderItem_refersTo_product)} REFERS_TO relationships...")
        for i in range(0, len(orderItem_refersTo_product), batch_size):
            batch = orderItem_refersTo_product[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (oi:Order_Item {order_item_id: rel[0].order_item_id})
                MATCH (p:Product {product_id: rel[1].product_id})
                MERGE (oi)-[:REFERS_TO]->(p)
                """,
                relationships=batch
            )
    
    # (OrderItem) –[:SOLD_BY]→ (Seller)
    if orderItem_soldBy_seller:
        print(f"Creating {len(orderItem_soldBy_seller)} SOLD_BY relationships...")
        for i in range(0, len(orderItem_soldBy_seller), batch_size):
            batch = orderItem_soldBy_seller[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (oi:Order_Item {order_item_id: rel[0].order_item_id})
                MATCH (s:Seller {seller_id: rel[1].seller_id})
                MERGE (oi)-[:SOLD_BY]->(s)
                """,
                relationships=batch
            )
    
    # (Customer)–[:WROTE]→ (Review)
    if customer_wrote_review:
        print(f"Creating {len(customer_wrote_review)} WROTE relationships...")
        for i in range(0, len(customer_wrote_review), batch_size):
            batch = customer_wrote_review[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (c:Customer {customer_id: rel[0].customer_id})
                MATCH (r:Review {review_id: rel[1].review_id})
                MERGE (c)-[:WROTE]->(r)
                """,
                relationships=batch
            )
    
    # (Review)–[:REVIEWS]→ (Order)
    if review_reviews_order:
        print(f"Creating {len(review_reviews_order)} REVIEWS relationships...")
        for i in range(0, len(review_reviews_order), batch_size):
            batch = review_reviews_order[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (r:Review {review_id: rel[0].review_id})
                MATCH (o:Order {order_id: rel[1].order_id})
                MERGE (r)-[:REVIEWS]->(o)
                """,
                relationships=batch
            )
    
    # (Customer)-[:LOCATED_IN]→(State)
    if customer_locatedIn_state:
        print(f"Creating {len(customer_locatedIn_state)} LOCATED_IN relationships...")
        for i in range(0, len(customer_locatedIn_state), batch_size):
            batch = customer_locatedIn_state[i:i+batch_size]
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (c:Customer {customer_id: rel[0].customer_id})
                MATCH (st:State {name: rel[1].customer_state})
                MERGE (c)-[:LOCATED_IN]->(st)
                """,
                relationships=batch
            )
    
    return "All relationships created successfully"


# Initialize the driver
uri = 'neo4j+s://df998545.databases.neo4j.io'
user = "neo4j"
password = 'P49OO43v3OLd-Gu3PieJtAQw1ifPxlyxU9JUMq7H5MY'

driver = GraphDatabase.driver(uri, auth=(user, password))

# Build KG in two phases
print("=" * 80)
print("BUILDING KNOWLEDGE GRAPH - PHASE 1: NODES")
print("=" * 80)
with driver.session() as session:
    result = session.execute_write(build_kg_nodes)
    print(result)


print("\n" + "=" * 80)
print("BUILDING KNOWLEDGE GRAPH - PHASE 2: RELATIONSHIPS")
print("=" * 80)
with driver.session() as session:
    result = session.execute_write(build_kg_relationships)
    print(result)

print("\n" + "=" * 80)
print("Knowledge Graph built successfully!")
print("=" * 80)