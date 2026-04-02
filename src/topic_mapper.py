import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_topic_map(chunks, k=8):
    embs = model.encode(chunks)
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(embs)

    topics = {}
    for i, label in enumerate(kmeans.labels_):
        topics.setdefault(label, []).append(chunks[i])
    return topics
