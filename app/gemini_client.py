"""Shared Gemini client, used by both embeddings.py and llm.py."""

import os

from google import genai
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ["GEMINI_API_KEY"]
        _client = genai.Client(api_key=api_key)
    return _client


def _is_rate_limit_error(exception: BaseException) -> bool:
    return isinstance(exception, APIError) and exception.code == 429


# Gemini's free tier caps generate_content at a handful of requests/minute (shared
# across image captioning and chat generation) — this backs off and retries rather
# than crashing ingestion or a live chat request the moment that's hit.
retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=15, min=15, max=65),
    stop=stop_after_attempt(5),
    reraise=True,
)
