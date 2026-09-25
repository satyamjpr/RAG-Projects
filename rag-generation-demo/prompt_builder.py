def build_rag_prompt(context, question):
    """
    Build a prompt that instructs an LLM to answer using retrieved context.

    The prompt explicitly tells the model not to rely on information
    outside the provided context and to avoid guessing when the answer
    cannot be found.

    Args:
        context: Relevant information retrieved from the documents.
        question: The user's question.

    Returns:
        A formatted prompt containing instructions, context, and question.

    Raises:
        ValueError: If context or question is empty.
    """
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return f"""Answer the question using only the information provided in the context.

Do not use outside knowledge or make assumptions.
If the answer cannot be found in the context, say that the information is not available.

Context:
{context}

Question:
{question}

Answer:"""