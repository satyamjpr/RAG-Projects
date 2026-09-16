from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(first_embedding, second_embedding):
    """
    Calculate the cosine similarity between two embeddings.

    Args:
        first_embedding: The first sentence embedding.
        second_embedding: The second sentence embedding.

    Returns:
        The cosine similarity score.
    """
    return cosine_similarity([first_embedding], [second_embedding])[0][0]


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love developing Laravel applications.",
    "I enjoy building applications with Laravel.",
    "I like cooking Indian food.",
]

embeddings = model.encode(sentences)

similarity_laravel = calculate_similarity(embeddings[0], embeddings[1])
similarity_cooking = calculate_similarity(embeddings[0], embeddings[2])

print(f"Laravel vs Laravel: {similarity_laravel:.4f}")
print(f"Laravel vs Cooking: {similarity_cooking:.4f}")