"""
Compare Top-K retrieval results for a broad question.

The experiment compares how many relevant chunks are retrieved
when using different Top-K values.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieval import retrieve_top_k_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "sample_document.txt"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    """Run the Top-K comparison experiment."""

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        chunks = [
            line.strip()
            for line in file
            if line.strip()
        ]

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(chunks)

    question = "What are the company's remote work rules?"

    print(f"\nQuestion: {question}")

    for top_k in [1, 3]:
        results = retrieve_top_k_chunks(
            question,
            chunks,
            chunk_embeddings,
            model,
            top_k=top_k,
        )

        print(f"\n{'=' * 60}")
        print(f"Results with top_k={top_k}")
        print(f"{'=' * 60}")

        for rank, (chunk, score) in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Similarity: {score:.4f}")
            print(f"Chunk: {chunk}")


if __name__ == "__main__":
    main()