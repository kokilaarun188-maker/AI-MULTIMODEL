import io
from typing import List

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF's bytes using pypdf."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # Skip pages that fail to parse rather than failing the whole upload
            pages_text.append("")
    return "\n".join(pages_text).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    A simple character-based sliding window. Good enough for RAG chunking
    without pulling in a heavier text-splitting library.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap

    return chunks


def batched(items: list, batch_size: int):
    """Yield successive batches of `batch_size` from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
