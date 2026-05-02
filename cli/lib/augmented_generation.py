import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def generate_answer(search_results, query, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag(query, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }


def rag_command(query):
    return rag(query)
