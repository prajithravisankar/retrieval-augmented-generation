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


def generate_summary(search_results, query, limit):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {context}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag(query, summarize: bool = False, limit=DEFAULT_SEARCH_LIMIT):
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

    if summarize:
        llm_summary = generate_summary(search_results, query, limit)
        return {
            "query": query,
            "search_results": search_results[:limit],
            "answer": "",
            "llm_summary": llm_summary,
        }

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": generate_answer(search_results, query, limit),
        "llm_summary": "",
    }


def rag_command(query):
    return rag(query)


def summarize(query, limit=5):
    return rag(query=query, summarize=True, limit=limit)
