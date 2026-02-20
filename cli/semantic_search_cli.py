#!/usr/bin/env python3

import argparse

from lib.search_utils import load_movies
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="calls the search method with query and optional limit"
    )
    search_parser.add_argument("query", type=str, help="query to search for")
    search_parser.add_argument(
        "--limit", type=int, help="number of results to return (optional)"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_instance = SemanticSearch()
            documents = load_movies()
            search_instance.load_or_create_embeddings(documents)
            results = search_instance.search(args.query, args.limit)
            for i, res in enumerate(results):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['description']}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
