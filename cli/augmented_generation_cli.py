import argparse
from lib.hybrid_search import rrf_search_command

from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="search query for rag")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rrf_search_results = rrf_search_command(query=query, limit=5)
            llm_answer = prompt_llm(query, rrf_search_results)
            print("Search Results:")
            for result in rrf_search_results["results"]:
                print(f"- {result['title']}")

            print("\nRAG Response:")
            print(llm_answer)
        case _:
            parser.print_help()


def prompt_llm(query, rrf_search_results):
    docs = rrf_search_results["results"]
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected


if __name__ == "__main__":
    main()
