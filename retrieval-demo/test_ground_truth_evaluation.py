"""
Test retrieval quality using ground-truth chunk indexes.
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
    """Run ground-truth retrieval evaluation."""

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        chunks = [
            line.strip()
            for line in file
            if line.strip()
        ]

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(chunks)

    results = evaluate_retrieval_with_ground_truth(
        ground_truth_dataset,
        chunks,
        chunk_embeddings,
        model,
        top_k=3,
    )

    print("\nGround-Truth Retrieval Evaluation Results:")

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for result in results:
        print("\nQuestion:", result["question"])
        print("Retrieved Chunks:", result["retrieved_count"])
        print("True Positives:", result["true_positives"])
        print("False Positives:", result["false_positives"])
        print("False Negatives:", result["false_negatives"])
        print(f"Precision: {result['precision']:.2f}")
        print(f"Recall: {result['recall']:.2f}")
        print(f"F1 Score: {result['f1_score']:.2f}")
        print("-" * 60)

        total_precision += result["precision"]
        total_recall += result["recall"]
        total_f1 += result["f1_score"]

    count = len(results)

    print("\nOverall Evaluation:")
    print(f"Average Precision: {total_precision / count:.2f}")
    print(f"Average Recall: {total_recall / count:.2f}")
    print(f"Average F1 Score: {total_f1 / count:.2f}")


if __name__ == "__main__":
    main()