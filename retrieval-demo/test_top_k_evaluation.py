"""
Compare retrieval evaluation metrics across different Top-K values.
"""

from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval-demo"

sys.path.insert(0, str(RETRIEVAL_DIR))

from ground_truth_dataset import ground_truth_dataset
from retrieval import evaluate_retrieval_with_ground_truth


DOCUMENT_PATH = PROJECT_ROOT / "sample_document.txt"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    """Compare retrieval metrics for different Top-K values."""

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        chunks = [
            line.strip()
            for line in file
            if line.strip()
        ]

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(chunks)

    print("\nTop-K Retrieval Evaluation")
    print("=" * 70)

    for top_k in [1, 2, 3]:

        results = evaluate_retrieval_with_ground_truth(
            ground_truth_dataset,
            chunks,
            chunk_embeddings,
            model,
            top_k=top_k,
        )

        total_precision = sum(
            result["precision"]
            for result in results
        )

        total_recall = sum(
            result["recall"]
            for result in results
        )

        total_f1 = sum(
            result["f1_score"]
            for result in results
        )

        count = len(results)

        average_precision = total_precision / count
        average_recall = total_recall / count
        average_f1 = total_f1 / count

        print(f"\nTop-K: {top_k}")
        print(f"Average Precision: {average_precision:.2f}")
        print(f"Average Recall:    {average_recall:.2f}")
        print(f"Average F1 Score:  {average_f1:.2f}")


if __name__ == "__main__":
    main()