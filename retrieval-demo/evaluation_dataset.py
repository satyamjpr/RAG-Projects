"""
Sample evaluation dataset for testing RAG retrieval quality.

Each test case defines the expected relevant document chunks.
"""

evaluation_dataset = [
    {
        "question": "How many days can employees work from home?",
        "expected_keywords": ["three days per week"],
    },
    {
        "question": "Who should employees inform before working remotely?",
        "expected_keywords": ["inform their manager"],
    },
    {
        "question": "How should remote work requests be submitted?",
        "expected_keywords": ["HR portal"],
    },
    {
        "question": "What should employees avoid storing on personal devices?",
        "expected_keywords": ["sensitive company information"],
    },
]