import argparse
from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="generate an embedding for an image and print its shape",
    )
    verify_parser.add_argument("image_path", help="path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)


if __name__ == "__main__":
    main()
