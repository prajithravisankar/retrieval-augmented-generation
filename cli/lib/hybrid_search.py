import os

from lib.keyword_search import InvertedIndex
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        raw_keyword_scores = [doc["score"] for doc in keyword_results]
        raw_semantic_scores = [doc["score"] for doc in semantic_results]

        normalized_keyword = (
            normalize_scores(raw_keyword_scores) if raw_keyword_scores else []
        )
        normalized_semantic = (
            normalize_scores(raw_semantic_scores) if raw_semantic_scores else []
        )

        master_tracker = {}

        for doc, norm_score in zip(keyword_results, normalized_keyword):
            doc_id = doc["id"]
            master_tracker[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc["document"],
                "keyword_score": norm_score,
                "semantic_score": 0.0,  # Default fallback
            }

        for doc, norm_score in zip(semantic_results, normalized_semantic):
            doc_id = doc["id"]
            if doc_id in master_tracker:
                master_tracker[doc_id]["semantic_score"] = norm_score
            else:
                master_tracker[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["document"],
                    "keyword_score": 0.0,  # Default fallback
                    "semantic_score": norm_score,
                }

        final_results = []
        for data in master_tracker.values():
            h_score = self.hybrid_score(
                data["keyword_score"], data["semantic_score"], alpha
            )
            data["hybrid_score"] = h_score
            final_results.append(data)

        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return final_results[:limit]

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


if __name__ == "__main__":
    from lib.search_utils import load_movies

    print("loading movies...")
    docs = load_movies()
    print("initializing hybrid search...")
    searcher = HybridSearch(docs)
    print("running weighted search...")
    searcher.weighted_search("some random query", 0.5)
