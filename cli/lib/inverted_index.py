import json
import os
import pickle


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        # 1. Simple tokenization (matches assignment expectations better)
        tokens = text.lower().split()

        for token in tokens:
            # 2. Fix the KeyError risk
            if token not in self.index:
                self.index[token] = set()

            # 3. Map Token -> Doc_ID (The Inversion)
            self.index[token].add(doc_id)

    def get_document(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def build(self):
        with open("data/movies.json", "r") as f:
            movies = json.load(f)
            for movie in movies:
                doc_id = movie["id"]
                text = f"{movie['title']} {movie['description']}"
                self.__add_document(doc_id, text)
                self.docmap[doc_id] = movie

    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
