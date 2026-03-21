import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="normalize a list of score"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case _:
            parser.print_help()


def normalize_command(scores):
    if not scores:
        return None
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        for _ in scores:
            print(f"* {1.0:.4f}")
        return

    for score in scores:
        normalized = (score - min_score) / (max_score - min_score)
        print(f"* {normalized:.4f}")


if __name__ == "__main__":
    main()
