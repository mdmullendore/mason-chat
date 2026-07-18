"""Builds a RAG prompt from retrieved chunks and streams the Groq response.

Uses Groq instead of Gemini for live chat generation — same reasoning as
app/vision.py: Groq's free tier (30 RPM / 1,000 RPD) has far more headroom than
Gemini's (5 RPM / 20 RPD) for this generation-heavy call under real traffic.
"""

import os
from collections.abc import Iterator

from app.groq_client import get_client, retry_on_rate_limit

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_system_prompt(chunks: list[dict]) -> str:
    context = "\n\n".join(f"- {chunk['text']}" for chunk in chunks)
    return (
        "You are a helpful assistant answering visitor questions about Mason Mullendore, "
        "on his personal website. Answer only using the context below. If the context "
        "doesn't contain the answer, say you don't have that information rather than "
        "guessing. Keep answers conversational and brief.\n\n"
        f"Context about Mason:\n{context}"
    )


@retry_on_rate_limit
def _create_stream(question: str, system_prompt: str):
    return get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )


def stream_answer(question: str, chunks: list[dict]) -> Iterator[str]:
    system_prompt = build_system_prompt(chunks)
    response = _create_stream(question, system_prompt)
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
