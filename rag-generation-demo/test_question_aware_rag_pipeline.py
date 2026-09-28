import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval-demo"

sys.path.insert(0, str(RETRIEVAL_DIR))


from rag_pipeline import run_question_aware_rag_pipeline


with open(PROJECT_ROOT / "sample_document.txt", "r", encoding="utf-8") as file:
    document = file.read()


chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]


model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)


question = "How many days can employees work from home?"

answer = run_question_aware_rag_pipeline(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
    min_similarity=0.4,
)

print("Question:")
print(question)

print("\nGenerated Answer:")
print(answer)


unknown_question = "What is the company's maternity leave policy?"

answer = run_question_aware_rag_pipeline(
    unknown_question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
    min_similarity=0.4,
)

print("\nUnknown Question:")
print(unknown_question)

print("\nGenerated Answer:")
print(answer)