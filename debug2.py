from hyb_retrive import bm25_retrieve, dense_retrieve

query = "Tell me about Dev's leadership experience"

print("Dense alone (no threshold, k=10):")
for doc, meta, dist in dense_retrieve(query, k=10, min_similarity=0.0):
    print(f"  {meta['title']}  sim={1-dist:.4f}")

print("\nBM25 alone:")
for cid, doc, meta, score in bm25_retrieve(query, k=10):
    print(f"  {meta['title']}  score={score:.4f}")