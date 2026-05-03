#! /usr/bin/env python3

import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verif_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verif_parser.add_argument("image", type=str, help="Path to image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="multimodal search with image"
    )
    image_search_parser.add_argument("image", help="path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for i, r in enumerate(results, start=1):
                desc = r["description"][:100] + (
                    "..." if len(r["description"]) > 100 else ""
                )
                print(f"{i}. {r['title']} (similarity: {r['similarity']:.3f})")
                print(f"   {desc}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
