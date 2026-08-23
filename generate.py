import os
from groq import Groq
from hybrid_retrieve import hybrid_retrieve

client = Groq(api_key=os.environ["GROQ_API_KEY"])  # set this in your shell / .env

SYSTEM_PROMPT = """You are an assistant answering questions about Dev Garg's resume, based ONLY on the context provided below.

Rules:
- Only use facts explicitly stated in the context. Do not add outside knowledge, even if you know it.
- If the answer is not present in the context, say clearly: "That's not something covered in Dev's resume."
- Do not guess or infer facts that aren't stated.
- When useful, mention which section the info came from (e.g. "According to the Projects section...").
- Be concise and direct, like a factual Q&A, not a sales pitch."""

def build_prompt(query: str, retrieved_chunks):
    context_blocks = []
    for doc, meta, score in retrieved_chunks:
        context_blocks.append(f"[{meta['section']} — {meta['title']}]\n{doc}")
    context = "\n\n".join(context_blocks)

    user_prompt = f"""Context:
{context}

Question: {query}"""
    return user_prompt

def answer(query: str, k: int = 3, min_similarity_gate: float = 0.55):
    retrieved = hybrid_retrieve(query, k=k, candidate_k=6)

    # Out-of-scope guard: if literally nothing relevant was retrieved
    # (e.g. all candidates got filtered by dense threshold, empty list),
    # don't even call the LLM with empty/junk context.
    if not retrieved:
        return "That's not something covered in Dev's resume."

    user_prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temp: factual Q&A, not creative writing
        max_tokens=300,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_queries = [
        "What computer vision projects has Dev worked on?",
        "What is Dev's CGPA?",
        "Tell me about Dev's leadership experience",
        "What is Dev's favorite food?",
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}")