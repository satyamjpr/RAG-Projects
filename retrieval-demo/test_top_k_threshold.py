
"""
Compare Top-K retrieval with and without a similarity threshold.
"""

from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval-demo"

sys.path.insert(0, str(RETRIEVAL_DIR))

from retrieval import (
    retrieve_top_k_chunks,
    retrieve_top_k_with_threshold,
)


DOCUMENT_PATH = PROJECT_ROOT / "sample_document.txt"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    """Run a Top-K and similarity-threshold comparison."""

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        chunks = [
            line.strip()
            for line in file
            if line.strip()
        ]

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(chunks)

    question = "What are the company's remote work rules?"
    top_k = 5
    threshold = 0.60

    print(f"\nQuestion: {question}")
    print(f"Top-K limit: {top_k}")
    print(f"Similarity threshold: {threshold}")

    regular_results = retrieve_top_k_chunks(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k=top_k,
    )

    filtered_results = retrieve_top_k_with_threshold(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k=top_k,
        threshold=threshold,
    )

    print("\n--- Standard Top-K Results ---")

    for rank, (chunk, score) in enumerate(regular_results, start=1):
        print(f"{rank}. Score: {score:.4f} | {chunk}")

    print("\n--- Top-K + Threshold Results ---")

    if not filtered_results:
        print("No chunks met the similarity threshold.")

    for rank, (chunk, score) in enumerate(filtered_results, start=1):
        print(f"{rank}. Score: {score:.4f} | {chunk}")

    print("\nSummary:")
    print(f"Standard Top-K returned: {len(regular_results)} chunks")
    print(f"After threshold filtering: {len(filtered_results)} chunks")


if __name__ == "__main__":
    main()