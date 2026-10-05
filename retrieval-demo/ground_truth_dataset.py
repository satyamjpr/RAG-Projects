"""
Ground-truth dataset for evaluating retrieval quality.

Each test case identifies the document chunks that are considered
relevant to the question.
"""

ground_truth_dataset = [
    {
        "question": "How many days can employees work from home?",
        "relevant_chunks": [0],
    },
    {
        "question": "Who should employees inform before working remotely?",
        "relevant_chunks": [1],
    },
    {
        "question": "How should remote work requests be submitted?",
        "relevant_chunks": [2],
    },
    {
        "question": "What should employees avoid storing on personal devices?",
        "relevant_chunks": [6],
    },
]