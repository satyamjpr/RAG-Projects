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


def evaluate_retrieval_result(
    retrieved_chunks,
    expected_keyword,
):
    """
    Check whether retrieved chunks contain the expected information.

    Args:
        retrieved_chunks: Retrieved chunks with similarity scores.
        expected_keyword: A keyword expected to appear in relevant content.

    Returns:
        True if the expected keyword is found in any retrieved chunk,
        otherwise False.
    """
    keyword = expected_keyword.lower()

    return any(
        keyword in chunk.lower()
        for chunk, _ in retrieved_chunks
    )


def evaluate_retrieval_dataset(
    evaluation_dataset,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
):
    """
    Evaluate retrieval performance using a collection of test questions.

    Each test case contains a question and an expected keyword.
    The function retrieves relevant chunks and checks whether the
    expected information appears in the retrieved results.

    Args:
        evaluation_dataset: List of test cases containing questions
            and their expected keywords.
        chunks: Document chunks available for retrieval.
        chunk_embeddings: Embeddings generated for document chunks.
        model: The sentence-transformer model used for retrieval.
        top_k: Number of chunks to retrieve for each question.

    Returns:
        A list of evaluation results containing each question,
        expected keyword, and whether the information was retrieved.
    """
    evaluation_results = []

    for test_case in evaluation_dataset:
        question = test_case["question"]
        expected_keyword = test_case["expected_keyword"]

        retrieved_chunks = retrieve_top_k_chunks(
            question,
            chunks,
            chunk_embeddings,
            model,
            top_k,
        )

        is_relevant = any(
            expected_keyword.lower() in chunk.lower()
            for chunk, _ in retrieved_chunks
        )

        evaluation_results.append({
            "question": question,
            "expected_keyword": expected_keyword,
            "retrieved": is_relevant,
        })

    return evaluation_results