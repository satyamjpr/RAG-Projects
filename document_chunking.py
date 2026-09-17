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
    Create overlapping chunks from complete sentences.

    Args:
        sentences: A list of sentences.
        sentences_per_chunk: Number of sentences in each chunk.
        overlap_sentences: Number of sentences shared between chunks.

    Returns:
        A list of text chunks.
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


with open("sample_document.txt", "r", encoding="utf-8") as file:
    document = file.read()

sentences = split_into_sentences(document)

chunks = create_chunks(
    sentences,
    sentences_per_chunk=2,
    overlap_sentences=1,
)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)