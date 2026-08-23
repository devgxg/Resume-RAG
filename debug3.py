# debug4.py
from hyb_retrive import hybrid_retrieve

query = "What computer vision projects has Dev worked on?"

candidates = hybrid_retrieve(query, k=6, candidate_k=6)
print(f"Candidates going into reranker ({len(candidates)} total):")
for doc, meta, score in candidates:
    print(f"  {meta['title']}  rrf_score={score:.4f}")