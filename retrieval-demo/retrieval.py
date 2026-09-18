from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunk(question, chunks, chunk_embeddings, model):
    """
    Retrieve the chunk most relevant to the given question.

    Args:
        question: The user's question.
        chunks: A list of document chunks.
        chunk_embeddings: Embeddings generated for the chunks.
        model: The sentence-transformer model.

    Returns:
        A tuple containing the most relevant chunk and its similarity score.
    """
    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings,
    )[0]

    best_chunk_index = similarities.argmax()

    return chunks[best_chunk_index], similarities[best_chunk_index]