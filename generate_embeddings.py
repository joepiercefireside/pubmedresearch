import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Database connection
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fetch articles
cur.execute("SELECT id, title, abstract FROM articles")
articles = cur.fetchall()

# Generate embeddings
for article in articles:
    id, title, abstract = article
    text = f"{title} {abstract or ''}".strip()
    if text:
        embedding = model.encode(text, convert_to_numpy=True)
        cur.execute(
            "UPDATE articles SET embedding = %s WHERE id = %s",
            (embedding.tolist(), id)
        )
    else:
        cur.execute(
            "UPDATE articles SET embedding = NULL WHERE id = %s",
            (id,)
        )
    conn.commit()

cur.close()
conn.close()