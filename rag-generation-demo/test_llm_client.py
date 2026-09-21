from llm_client import generate_answer


prompt = "What is Retrieval-Augmented Generation? Explain it in one simple sentence."

answer = generate_answer(prompt)

print("LLM Answer:")
print(answer)