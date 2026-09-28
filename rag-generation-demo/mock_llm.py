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


def generate_context_grounded_answer(prompt):
    """
    Generate an answer using information found in the provided RAG prompt.

    This function simulates a context-grounded LLM response without
    making an external API call.

    Args:
        prompt: A RAG prompt containing context and a question.

    Returns:
        An answer based only on supported information in the prompt.

    Raises:
        ValueError: If the prompt is empty.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    prompt_lower = prompt.lower()

    if "three days per week" in prompt_lower:
        return "Remote employees can work from home up to three days per week."

    if "manager before working remotely" in prompt_lower:
        return "Employees must inform their manager before working remotely."

    return "The answer could not be found in the provided context."


def generate_question_aware_answer(prompt):
    """
    Generate an answer by matching the user's question with relevant
    information available in the provided context.

    This function simulates a question-aware LLM response without making
    an external API call. The context and question are processed separately
    so that unrelated information in the context is not returned as the
    answer.

    Args:
        prompt: A RAG prompt containing context and a question.

    Returns:
        An answer based on information relevant to the question.

    Raises:
        ValueError: If the prompt is empty.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    prompt_lower = prompt.lower()

    context_start = prompt_lower.find("context:")
    question_start = prompt_lower.find("question:")

    if context_start == -1 or question_start == -1:
        return "The answer could not be found in the provided context."

    context = prompt_lower[
        context_start + len("context:"):question_start
    ]

    question = prompt_lower[
        question_start + len("question:"):
    ]

    if (
        "how many days" in question
        and "work from home" in question
        and "three days per week" in context
    ):
        return "Remote employees can work from home up to three days per week."

    if (
        "who" in question
        and "working remotely" in question
        and "manager before working remotely" in context
    ):
        return "Employees must inform their manager before working remotely."

    return "The answer could not be found in the provided context."