#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    semantic_search,
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
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="fixed size chunking to split long text into smaller pieces for embeddings.",
    )
    chunk_parser.add_argument("text", help="text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="optional argument for chunk size defaults to 200 words",
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
            semantic_search(args.query, args.limit)
        case "chunk":
            words = args.text.split()
            chunks = []
            for i in range(0, len(words), args.chunk_size):
                chunk_words = words[i : i + args.chunk_size]
                chunks.append(" ".join(chunk_words))
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
