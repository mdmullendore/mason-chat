"""Turns an image into a text caption at ingestion time, so it can be embedded and
retrieved through the same text pipeline as everything else (no separate image vector
space to reason about). Uses Groq's vision model — far more generous free-tier
limits (30 RPM / 1,000 RPD) than Gemini's (5 RPM / 20 RPD), which is what
image-heavy ingestion runs were hitting."""

import base64
import io

import pillow_heif
from PIL import Image

from app.groq_client import get_client, retry_on_rate_limit

pillow_heif.register_heif_opener()

VISION_MODEL = "qwen/qwen3.6-27b"

CAPTION_PROMPT = (
    "Describe this image factually and specifically, in 2-4 sentences, as if writing "
    "a caption for someone who can't see it. Mention concrete details (setting, "
    "people, objects, text visible in the image) rather than vague impressions. This "
    "will be used as searchable context about Mason Mullendore, so be specific enough "
    "that someone could tell what the image shows just from your description."
)

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _normalize_to_jpeg(image_bytes: bytes) -> bytes:
    """Re-encodes as a real JPEG regardless of the file's actual format — several
    of Mason's photos are HEIC (iPhone's native format) saved with a misleading
    .png/.jpg extension. Gemini tolerated that; Groq's vision API validates the
    bytes strictly and rejects the mismatch, so every image goes through this
    regardless of what its extension claims."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@retry_on_rate_limit
def caption_image(image_bytes: bytes) -> str:
    normalized = _normalize_to_jpeg(image_bytes)
    encoded = base64.b64encode(normalized).decode("utf-8")
    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        reasoning_effort="none",  # plain captioning needs no reasoning trace
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()
