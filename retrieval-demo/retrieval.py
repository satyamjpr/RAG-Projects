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


def retrieve_top_k_chunks(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
):
    """
    Retrieve the most relevant chunks for a given question.

    Args:
        question: The user's question.
        chunks: A list of document chunks.
        chunk_embeddings: Embeddings generated for the chunks.
        model: The sentence-transformer model.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        A list of tuples containing the chunk and similarity score.
    """
    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings,
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    return [
        (chunks[index], similarities[index])
        for index in ranked_indices
    ]


def retrieve_with_threshold(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
    min_similarity=0.4,
):
    """
    Retrieve relevant chunks only when their similarity meets the threshold.

    Args:
        question: The user's question.
        chunks: A list of document chunks.
        chunk_embeddings: Embeddings generated for the document chunks.
        model: The sentence-transformer model.
        top_k: Maximum number of chunks to retrieve.
        min_similarity: Minimum similarity score required for a chunk
            to be considered relevant.

    Returns:
        A list of relevant chunks with their similarity scores.
    """
    retrieved_chunks = retrieve_top_k_chunks(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k,
    )

    return [
        (chunk, score)
        for chunk, score in retrieved_chunks
        if score >= min_similarity
    ]