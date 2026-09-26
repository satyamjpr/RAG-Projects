import re
import sys

sys.path.append("../retrieval-demo")

from sentence_transformers import SentenceTransformer

from rag_pipeline import run_threshold_rag_pipeline


with open("../sample_document.txt", "r", encoding="utf-8") as file:
    document = file.read()


sentences = re.split(r"(?<=[.!?])\s+", document.strip())

chunks = []

for index in range(0, len(sentences), 2):
    chunk = " ".join(sentences[index:index + 2])
    chunks.append(chunk)


model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)


question = "What is the company's maternity leave policy?"

answer = run_threshold_rag_pipeline(
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