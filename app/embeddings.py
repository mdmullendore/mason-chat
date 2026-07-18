"""Gemini embedding wrapper, shared by the ingestion script and runtime query embedding.

gemini-embedding-2 doesn't take a task_type config param (unlike the now-deprecated
gemini-embedding-001) — the task is specified via an instruction prefix in the text
itself, per Google's recommended format. It also doesn't reliably batch multiple texts
in one `contents` list (observed: 3 texts in, 1 embedding back, no error) — so each text
is embedded with its own call.
"""

from google.genai import types

from app.gemini_client import get_client

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def _format_document(text: str, source: str) -> str:
    return f"title: {source} | text: {text}"


def _format_query(text: str) -> str:
    return f"task: search result | query: {text}"


def embed_documents(chunks: list[dict]) -> list[list[float]]:
    """chunks: [{"text": ..., "source": ...}, ...] — used at ingestion time."""
    return [_embed_one(_format_document(c["text"], c["source"])) for c in chunks]


def embed_query(text: str) -> list[float]:
    """Used at request time to embed the incoming question."""
    return _embed_one(_format_query(text))


def _embed_one(formatted_text: str) -> list[float]:
    result = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=formatted_text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return result.embeddings[0].values
