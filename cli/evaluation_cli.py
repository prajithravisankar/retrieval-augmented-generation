import argparse
import json
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evalutate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        d = json.load(f)

    for test_case in d["test_cases"]:
        query = test_case["query"]
        golden_titles = test_case["relevant_docs"]

        rrf_search_result = rrf_search_command(query=query, k=60, limit=limit)

        retrieved_titles = []
        for doc in rrf_search_result["results"]:
            retrieved_titles.append(doc["title"])

        common = 0
        total = len(retrieved_titles)
        for title in retrieved_titles:
            if title in golden_titles:
                common += 1

        precision = common / total if total > 0 else 0.0
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(golden_titles)}\n")


if __name__ == "__main__":
    main()
