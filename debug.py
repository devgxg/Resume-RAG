# debug.py
from hyb_retrive import bm25_retrieve, dense_retrieve

query = "What is Dev's CGPA?"

print("BM25 alone:")
for cid, doc, meta, score in bm25_retrieve(query, k=5):
    print(f"  [{score:.4f}] {meta['title']}")

print("\nDense alone:")
for doc, meta, dist in dense_retrieve(query, k=5):
    print(f"  [{1-dist:.4f}] {meta['title']}")