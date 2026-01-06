import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)

def build_memory(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks
