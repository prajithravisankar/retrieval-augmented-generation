import os

from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])  # type: ignore[arg-type]
        return image_embedding[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        results = []
        for doc, text_embedding in zip(self.documents, self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": score,
                }
            )
        results.sort(key=lambda result: result["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    movies = load_movies()
    searcher = MultimodalSearch(documents=movies)
    return searcher.search_with_image(image_path=image_path)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
