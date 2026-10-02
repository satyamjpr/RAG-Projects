import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "retrieval-demo"))

from evaluation_dataset import evaluation_dataset
from retrieval import evaluate_retrieval_dataset


with open(
    PROJECT_ROOT / "sample_document.txt",
    "r",
    encoding="utf-8",
) as file:
    document = file.read()


chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]


model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)


results = evaluate_retrieval_dataset(
    evaluation_dataset,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
)


print("Retrieval Evaluation Results:\n")

for result in results:
    print(f"Question: {result['question']}")
    print(f"Retrieved Chunks: {result['retrieved_count']}")
    print(f"Relevant Retrieved: {result['relevant_retrieved']}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall: {result['recall']:.2f}")
    print("-" * 60)


if results:
    average_precision = sum(
        result["precision"] for result in results
    ) / len(results)

    average_recall = sum(
        result["recall"] for result in results
    ) / len(results)

    print("\nOverall Evaluation:")
    print(f"Average Precision: {average_precision:.2f}")
    print(f"Average Recall: {average_recall:.2f}")