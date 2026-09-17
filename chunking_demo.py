import re


def split_into_sentences(text):
    """
    Split text into individual sentences.

    Args:
        text: The source text to split.

    Returns:
        A list of sentences.
    """
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def create_chunks(sentences, sentences_per_chunk=2, overlap_sentences=1):
    """
    Create chunks from complete sentences with configurable overlap.

    Args:
        sentences: A list of sentences.
        sentences_per_chunk: Number of sentences in each chunk.
        overlap_sentences: Number of sentences shared with the next chunk.

    Returns:
        A list of overlapping text chunks.
    """
    if overlap_sentences >= sentences_per_chunk:
        raise ValueError("Overlap must be smaller than the chunk size.")

    chunks = []
    step = sentences_per_chunk - overlap_sentences

    for start in range(0, len(sentences), step):
        chunk_sentences = sentences[start:start + sentences_per_chunk]

        if not chunk_sentences:
            break

        chunks.append(" ".join(chunk_sentences))

        if start + sentences_per_chunk >= len(sentences):
            break

    return chunks


text = """
Employees can work from home up to three days per week.
Employees must inform their manager before working remotely.
Remote work requests should be submitted through the HR portal.
Employees are responsible for maintaining a suitable work environment.
"""

sentences = split_into_sentences(text)
chunks = create_chunks(sentences, sentences_per_chunk=2, overlap_sentences=1)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)