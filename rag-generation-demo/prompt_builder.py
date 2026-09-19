def build_rag_prompt(context, question):
    """
    Build a prompt that provides retrieved context and the user's question.

    Args:
        context: Relevant information retrieved from the documents.
        question: The user's question.

    Returns:
        A prompt containing the context and question for the LLM.
    """
    return f"""Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""