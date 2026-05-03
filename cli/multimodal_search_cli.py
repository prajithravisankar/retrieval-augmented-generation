#! /usr/bin/env python3

import argparse

from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verif_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verif_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
