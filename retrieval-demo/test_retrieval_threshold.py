import re

from sentence_transformers import SentenceTransformer

from retrieval import retrieve_with_threshold


with open("../sample_document.txt", "r", encoding="utf-8") as file:
    document = file.read()


sentences = re.split(r"(?<=[.!?])\s+", document.strip())

chunks = []

for index in range(0, len(sentences), 2):
    chunk = " ".join(sentences[index:index + 2])
    chunks.append(chunk)


model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)


question = "What is the company's maternity leave policy?" # Unknown question
# question = "How many days can employees work from home?" # Known question

results = retrieve_with_threshold(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
    min_similarity=0.4,
)


print("Question:")
print(question)

print("\nRelevant Chunks:")

if not results:
    print("No sufficiently relevant information found.")
else:
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank} - Similarity: {score:.4f}")
        print(chunk)