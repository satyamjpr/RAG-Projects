from context_builder import build_context
from mock_llm import generate_mock_answer
from prompt_builder import build_rag_prompt
from retrieval import retrieve_top_k_chunks
from retrieval import retrieve_with_threshold


def run_rag_pipeline(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
):
    """
    Run the complete retrieval-augmented generation pipeline.

    The pipeline retrieves relevant document chunks, builds the context,
    creates a RAG prompt, and generates an answer using the mock LLM.

    Args:
        question: The user's question.
        chunks: A list of document chunks.
        chunk_embeddings: Embeddings generated for the document chunks.
        model: The sentence-transformer model used for retrieval.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        The generated answer.
    """
    retrieved_chunks = retrieve_top_k_chunks(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k,
    )

    context = build_context(retrieved_chunks)

    prompt = build_rag_prompt(
        context,
        question,
    )

    return generate_mock_answer(prompt)


def run_threshold_rag_pipeline(
    question,
    chunks,
    chunk_embeddings,
    model,
    top_k=3,
    min_similarity=0.4,
):
    """
    Run a RAG pipeline with similarity-based relevance filtering.

    The pipeline retrieves the most relevant chunks, removes chunks
    below the similarity threshold, builds the context, creates the
    RAG prompt, and generates an answer using the mock LLM.

    Args:
        question: The user's question.
        chunks: A list of document chunks.
        chunk_embeddings: Embeddings generated for the document chunks.
        model: The sentence-transformer model used for retrieval.
        top_k: Maximum number of chunks to retrieve.
        min_similarity: Minimum similarity score required for a chunk
            to be considered relevant.

    Returns:
        The generated answer, or a fallback message when no relevant
        information is found.
    """
    retrieved_chunks = retrieve_with_threshold(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k,
        min_similarity,
    )

    if not retrieved_chunks:
        return "The answer could not be found in the provided context."

    context = build_context(retrieved_chunks)

    prompt = build_rag_prompt(
        context,
        question,
    )

    return generate_mock_answer(prompt)