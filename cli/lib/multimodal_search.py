from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        img = Image.open(image_path)
        return self.model.encode([img])[0]  # type: ignore


def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path=image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
