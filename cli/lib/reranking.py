import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    doc_list_str = ""
    for doc in documents:
        doc_list_str += f"\nID: {doc.get('id')} - Title: {doc.get('title', '')} - {doc.get('document', '')}"

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.models.generate_content(model=model, contents=prompt)
    response_text = (response.text or "").strip()

    # --- THE NEW CLEANUP CODE ---
    # Strip markdown code blocks if the LLM added them
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    # Safely try to load the JSON, fallback to original order if it fails
    try:
        ranked_ids = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON. Raw response: {response_text}")
        return documents[:limit]
    # -----------------------------

    rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
    ranked_docs = []
    for doc in documents:
        doc_id = doc["id"]
        if doc_id in rank_map:
            ranked_docs.append({**doc, "batch_rank": rank_map[doc_id]})

    ranked_docs.sort(key=lambda x: x.get("batch_rank", 999))
    return ranked_docs[:limit]


def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    elif method == "batch":
        return llm_rerank_batch(query, documents, limit)
    else:
        return documents[:limit]
