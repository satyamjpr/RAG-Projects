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
    status = "PASS" if result["retrieved"] else "FAIL"

    print(f"Question: {result['question']}")
    print(f"Expected Keyword: {result['expected_keyword']}")
    print(f"Result: {status}")
    print("-" * 60)


passed = sum(
    1 for result in results
    if result["retrieved"]
)

total = len(results)

print(f"\nEvaluation Summary: {passed}/{total} questions passed.")