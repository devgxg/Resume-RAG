# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder
from hyb_retrive import hybrid_retrieve

# Standard, well-tested cross-encoder for reranking. Small, fast, CPU-friendly.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates, top_k: int = 6):
    """
    candidates: list of (doc, meta, score) -- from hybrid_retrieve, the fused RRF list.
    We ignore the RRF score entirely here; the cross-encoder judges fresh.
    """
    pairs = [[query, doc] for doc, meta, _ in candidates]
    ce_scores = cross_encoder.predict(pairs)  # one relevance score per (query, doc) pair

    scored = list(zip(candidates, ce_scores))
    scored.sort(key=lambda x: -x[1])  # highest relevance first

    return [(doc, meta, ce_score) for (doc, meta, _), ce_score in scored[:top_k]]

def retrieve_and_rerank(query: str, candidate_k: int = 10, final_k: int = 6):
    candidates = hybrid_retrieve(query, k=candidate_k, candidate_k=candidate_k)
    return rerank(query, candidates, top_k=final_k)

if __name__ == "__main__":
    test_queries = [
        "What computer vision projects has Dev worked on?",
        "What is Dev's CGPA?",
        "Tell me about Dev's leadership experience",
        "What is Dev's favorite food?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for doc, meta, score in retrieve_and_rerank(q):
            print(f"  [{score:.4f}] {meta['title']} ({meta['section']})")