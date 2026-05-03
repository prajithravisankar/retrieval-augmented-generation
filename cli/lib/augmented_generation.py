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


def generate_answer_with_citations(search_results, query, limit=5):
    context = ""

    for i, result in enumerate(search_results[:limit], start=1):
        context += f"[{i}]: {result['title']}; {result['document']}\n\n"

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {context}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)

    return (response.text or "").strip()


def multi_document_summary(search_results, query, limit=5):
    docs_text = ""
    for i, result in enumerate(search_results[:limit], start=1):
        docs_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {docs_text}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def answer_question(search_results, question, limit=5):
    context = ""

    for i, result in enumerate(search_results[:limit], start=1):
        context += f"[{i}]: {result['title']}; {result['document']}\n\n"

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

    Question: {question}

    Documents:
    {context}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

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


def summarize_command(query, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {"query": query, "error": "No results found"}

    summary = multi_document_summary(search_results, query, limit)

    return {
        "query": query,
        "summary": summary,
        "search_results": search_results[:limit],
    }


def citations_command(query, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {"query": query, "error": "No results found"}

    result = generate_answer_with_citations(search_results, query, limit)

    return {
        "query": query,
        "answer": result,
        "search_results": search_results,
    }


def question_command(question, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(question, k=RRF_K)

    if not search_results:
        return {"question": question, "error": "No results found"}

    result = answer_question(search_results, question, limit)

    return {
        "question": question,
        "answer": result,
        "search_results": search_results,
    }
