def build_context(retrieved_chunks):
    """
    Combine retrieved document chunks into a single context string.

    Args:
        retrieved_chunks: A list of retrieved chunks with similarity scores.

    Returns:
        A formatted string containing the retrieved context.
    """
    return "\n\n".join(
        chunk
        for chunk, _ in retrieved_chunks
    )