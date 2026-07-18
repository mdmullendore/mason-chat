"""Shared Groq client, used for image captioning and chat generation.

Groq doesn't offer an embeddings API, so app/embeddings.py stays on Gemini
(gemini_client.py) — only the two generation-heavy calls (vision captioning,
chat answers) moved here, since Groq's free tier is far more generous for those
than Gemini's (30 RPM / 1,000 RPD vs. Gemini's 5 RPM / 20 RPD for generate_content).
"""

import os

from groq import Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
