from context_builder import build_context
from mock_llm import generate_mock_answer
from prompt_builder import build_rag_prompt
from retrieval import retrieve_top_k_chunks


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