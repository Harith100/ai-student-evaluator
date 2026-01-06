def chunk_text(text: str, size: int = 600, overlap: int = 100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks
