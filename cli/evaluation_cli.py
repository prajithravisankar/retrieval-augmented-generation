import argparse

from lib.evaluation import evaluate_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    result = evaluate_command(args.limit)

    print(f"k={args.limit}\n")
    for query, res in result["results"].items():
        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {res['precision']:.4f}")
        print(f"  - Recall@{args.limit}: {res['recall']:.4f}")
        print(f"  - Retrieved: {', '.join(res['retrieved'])}")
        print(f"  - Relevant: {', '.join(res['relevant'])}")
        print()


if __name__ == "__main__":
    main()
