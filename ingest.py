import chromadb
from sentence_transformers import SentenceTransformer
from data import CHUNKS

# bge-small-en-v1.5: real transformer-based embedding model, 384-dim,
# strong retrieval quality for its size, runs fine on CPU.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Persistent client -> writes to disk in ./chroma_db, survives restarts.
# (Use chromadb.Client() instead for an in-memory-only version.)
client = chromadb.PersistentClient(path="./chroma_db")

# get_or_create so re-running this script doesn't crash on a duplicate collection.
collection = client.get_or_create_collection(
    name="resume_chunks",
    metadata={"hnsw:space": "cosine"}  # explicitly tell Chroma to use cosine similarity
)

texts = [c["text"] for c in CHUNKS]
ids = [c["id"] for c in CHUNKS]
metadatas = [{"section": c["section"], "title": c["title"]} for c in CHUNKS]

# normalize_embeddings=True -> vectors have unit length, so cosine similarity
# becomes a simple dot product internally (cheaper, same math we discussed).
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

# upsert = insert or update if id already exists. Safe to rerun this script.
collection.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
)

print(f"Ingested {collection.count()} chunks into ChromaDB.")