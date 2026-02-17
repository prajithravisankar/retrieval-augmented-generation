#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex, build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="gives frequency of the given term in the given document"
    )
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term to find in the document")

    args = parser.parse_args()

    match args.command:
        case "tf":
            try:
                print("Counting Term frequency in the given document...")
                index = InvertedIndex()
                index.load()
                count = index.get_tf(args.doc_id, args.term)
                print(count)
            except FileNotFoundError:
                print("Error: Index not found. Please run 'build' first.")
            except Exception as e:
                print(f"Error: {e}")

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
