"""
Sample evaluation dataset for testing RAG retrieval quality.

Each test case contains a question and the expected relevant
document information.
"""

evaluation_dataset = [
    {
        "question": "How many days can employees work from home?",
        "expected_keyword": "three days per week",
    },
    {
        "question": "Who should employees inform before working remotely?",
        "expected_keyword": "manager",
    },
    {
        "question": "How should remote work requests be submitted?",
        "expected_keyword": "HR portal",
    },
    {
        "question": "What should employees avoid storing on personal devices?",
        "expected_keyword": "sensitive company information",
    },
]