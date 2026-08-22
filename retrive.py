import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="resume_chunks",
    metadata={"hnsw:space": "cosine"}
)

def dense_retrieve(query: str, k: int = 3):
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    query_vec = model.encode([prefixed_query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=k,
    )

    # results is a dict of lists (Chroma's API returns batched results,
    # we only sent 1 query so we index [0] everywhere)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine DISTANCE (1 - similarity), lower = better

    return list(zip(docs, metadatas, distances))

if __name__ == "__main__":
    test_queries = [
        "What computer vision projects has Dev worked on?",
        "What is Dev's CGPA?",
        "Tell me about Dev's leadership experience",
        "What is Dev's favorite food?",  # deliberately out-of-scope
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        for doc, meta, dist in dense_retrieve(q, k=3):
            similarity = 1 - dist
            print(f"  [{similarity:.3f}] {meta['title']} ({meta['section']})")