# debug2.py
from hyb_retrive import bm25_retrieve, dense_retrieve

query = "What computer vision projects has Dev worked on?"

print("Dense candidates (post-threshold):")
for doc, meta, dist in dense_retrieve(query, k=5):
    print(f"  {meta['title']}  sim={1-dist:.4f}")

print("\nBM25 candidates:")
for cid, doc, meta, score in bm25_retrieve(query, k=5):
    print(f"  {meta['title']}  score={score:.4f}")