import os
import re
from dotenv import load_dotenv

load_dotenv()
from groq import Groq
from hyb_retrive import hybrid_retrieve

client = Groq(api_key=os.environ["GROQ_API_KEY"])

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
    return f"""Context:
{context}

Question: {query}"""


def strip_thinking(text: str) -> str:
    # Defensive: strips <think>...</think> if a reasoning model ever gets used again.
    # No-op (returns text unchanged) for non-thinking models like gemma2-9b-it.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def answer(query: str, k: int = 3):
    retrieved = hybrid_retrieve(query, k=k, candidate_k=6)

    if not retrieved:
        return "That's not something covered in Dev's resume."

    user_prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # non-thinking, fast, instruction-tuned -- fits this factual-lookup task
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return strip_thinking(response.choices[0].message.content)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "What computer vision projects has Dev worked on?",
        "What is Dev's CGPA?",
        "Tell me about Dev's leadership experience",
        "What is Dev's favorite food?",
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}")