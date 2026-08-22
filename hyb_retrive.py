from rank_bm25 import BM25Okapi
from retrive import dense_retrieve, collection
import re

# --- BM25 setup ---
# BM25 needs raw text tokenized, and works over the full corpus (not per-query),
# so we build the index once at import time.
all_data = collection.get()  # pulls everything back out of Chroma
bm25_ids = all_data["ids"]
bm25_docs = all_data["documents"]
bm25_metas = all_data["metadatas"]

STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "in", "on", "for", "what",
    "who", "how", "does", "has", "have", "with", "dev",  # "dev" appears in every
    # query by virtue of being the resume owner's name -- it's not a distinguishing
    # search term here, it's noise. Excluding it is a deliberate, documented choice.
}

def tokenize(text):
    text = text.replace("'s", "")  # strip possessive BEFORE splitting, not after
    tokens = re.findall(r"[a-z]+", text.lower())  # letters only, no stray single chars
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

tokenized_corpus = [tokenize(doc) for doc in bm25_docs]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_retrieve(query: str, k: int = 3):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(bm25_ids, bm25_docs, bm25_metas, scores), key=lambda x: -x[3])
    return ranked[:k]

# --- Reciprocal Rank Fusion ---
def rrf_fuse(dense_results, bm25_results, k_const: int = 60, top_k: int = 3):
    """
    dense_results: list of (doc, meta, distance) from dense_retrieve
    bm25_results:  list of (id, doc, meta, score) from bm25_retrieve
    RRF score for an item = sum over each list it appears in of 1 / (k_const + rank)
    k_const=60 is the standard default from the original RRF paper -- dampens the
    impact of rank 1 vs rank 2 so one list can't completely dominate the fusion.
    """
    scores = {}
    info = {}  # id -> (doc, meta) for final output

    for rank, (doc, meta, dist) in enumerate(dense_results):
        # we don't have chunk id here, use title as a stand-in unique key
        key = meta["title"]
        scores[key] = scores.get(key, 0) + 1 / (k_const + rank + 1)
        info[key] = (doc, meta)

    for rank, (cid, doc, meta, score) in enumerate(bm25_results):
        key = meta["title"]
        scores[key] = scores.get(key, 0) + 1 / (k_const + rank + 1)
        info[key] = (doc, meta)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [(info[key][0], info[key][1], score) for key, score in fused[:top_k]]

def hybrid_retrieve(query: str, k: int = 3, candidate_k: int = 5):
    dense = dense_retrieve(query, k=candidate_k)
    sparse = bm25_retrieve(query, k=candidate_k)
    return rrf_fuse(dense, sparse, top_k=k)

if __name__ == "__main__":
    test_queries = [
        "What computer vision projects has Dev worked on?",
        "What is Dev's CGPA?",
        "Tell me about Dev's leadership experience",
        "What is Dev's favorite food?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for doc, meta, score in hybrid_retrieve(q):
            print(f"  [{score:.4f}] {meta['title']} ({meta['section']})")