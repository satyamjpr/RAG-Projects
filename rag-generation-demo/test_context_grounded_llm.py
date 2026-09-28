from mock_llm import generate_context_grounded_answer


prompt = """Answer the question using only the information provided in the context.

Context:
Remote employees can work from home up to three days per week.
Employees must inform their manager before working remotely.

Question:
How many days can employees work from home?

Answer:"""


answer = generate_context_grounded_answer(prompt)

print("Generated Answer:")
print(answer)

unknown_prompt = """Answer the question using only the information provided in the context.

Context:
Employees must inform their manager before working remotely.

Question:
What is the company's maternity leave policy?

Answer:"""


unknown_answer = generate_context_grounded_answer(
    unknown_prompt
)

print("\nGenerated Answer for Unknown Question:")
print(unknown_answer)