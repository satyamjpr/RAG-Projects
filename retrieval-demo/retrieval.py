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

    Each test case contains a question and expected keywords.
    Retrieved chunks are checked against these keywords to calculate
    precision and recall for each question.

    Args:
        evaluation_dataset: List of test cases containing questions
            and their expected keywords.
        chunks: Document chunks available for retrieval.
        chunk_embeddings: Embeddings generated for document chunks.
        model: The sentence-transformer model used for retrieval.
        top_k: Number of chunks to retrieve for each question.

    Returns:
        A list of evaluation results containing retrieved chunk counts,
        relevant chunk counts, precision, and recall.
    """
    evaluation_results = []

    for test_case in evaluation_dataset:
        question = test_case["question"]
        expected_keywords = [
            keyword.lower()
            for keyword in test_case["expected_keywords"]
        ]

        retrieved_chunks = retrieve_top_k_chunks(
            question,
            chunks,
            chunk_embeddings,
            model,
            top_k,
        )

        relevant_chunks = [
            chunk
            for chunk, _ in retrieved_chunks
            if any(
                keyword in chunk.lower()
                for keyword in expected_keywords
            )
        ]

        relevant_retrieved = len(relevant_chunks)
        total_retrieved = len(retrieved_chunks)

        precision = (
            relevant_retrieved / total_retrieved
            if total_retrieved
            else 0.0
        )

        recall = (
            relevant_retrieved / len(expected_keywords)
            if expected_keywords
            else 0.0
        )

        evaluation_results.append({
            "question": question,
            "retrieved_count": total_retrieved,
            "relevant_retrieved": relevant_retrieved,
            "precision": precision,
            "recall": recall,
        })

    return evaluation_results


def evaluate_retrieval_with_ground_truth(
    evaluation_dataset,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
):
    """
    Evaluate retrieval quality using ground-truth chunk indexes.

    Precision measures how many retrieved chunks are relevant.
    Recall measures how many relevant chunks were successfully retrieved.
    F1 score combines precision and recall into a single metric.

    Args:
        evaluation_dataset: Test cases containing questions and relevant
            chunk indexes.
        chunks: Document chunks available for retrieval.
        chunk_embeddings: Embeddings generated for document chunks.
        model: The sentence-transformer model used for retrieval.
        top_k: Number of chunks to retrieve for each question.

    Returns:
        A list of evaluation results containing precision, recall,
        and F1 score for each question.
    """
    evaluation_results = []

    for test_case in evaluation_dataset:
        question = test_case["question"]
        relevant_chunk_indexes = set(test_case["relevant_chunks"])

        retrieved_chunks = retrieve_top_k_chunks(
            question,
            chunks,
            chunk_embeddings,
            model,
            top_k,
        )

        retrieved_indexes = {
            chunks.index(chunk)
            for chunk, _ in retrieved_chunks
        }

        true_positives = len(
            retrieved_indexes & relevant_chunk_indexes
        )

        false_positives = len(
            retrieved_indexes - relevant_chunk_indexes
        )

        false_negatives = len(
            relevant_chunk_indexes - retrieved_indexes
        )

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        evaluation_results.append({
            "question": question,
            "retrieved_count": len(retrieved_indexes),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        })

    return evaluation_results