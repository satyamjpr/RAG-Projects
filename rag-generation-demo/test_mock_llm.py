from mock_llm import generate_mock_answer


prompt = """Use the following context to answer the question.

Context:
Remote employees can work from home up to three days per week.

Question:
How many days can employees work from home?

Answer:"""


answer = generate_mock_answer(prompt)

print("Generated Answer:")
print(answer)

unknown_prompt = """Use the following context to answer the question.

Context:
Employees must inform their manager before working remotely.

Question:
What is the company's annual bonus?

Answer:"""


unknown_answer = generate_mock_answer(unknown_prompt)

print("\nGenerated Answer for Unknown Question:")
print(unknown_answer)