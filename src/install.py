from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

SentenceTransformer("all-MiniLM-L6-v2")
WhisperModel("small")
print("Models warmed.")