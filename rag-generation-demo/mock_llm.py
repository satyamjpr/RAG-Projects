def generate_mock_answer(prompt):
    """
    Generate a simple mock answer from information contained in a RAG prompt.

    This function simulates the final generation step of a RAG pipeline
    without making an external API call.

    Args:
        prompt: A RAG prompt containing context and a question.

    Returns:
        An answer based on the available context.

    Raises:
        ValueError: If the prompt is empty.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if "three days per week" in prompt.lower():
        return "Remote employees can work from home up to three days per week."

    return "The answer could not be found in the provided context."