from mock_llm import generate_question_aware_answer


known_prompt = """
Answer the question using only the information provided in the context.

Context:
Remote employees can work from home up to three days per week.
Employees must inform their manager before working remotely.

Question:
How many days can employees work from home?

Answer:
"""

answer = generate_question_aware_answer(known_prompt)

print("Generated Answer:")
print(answer)


unknown_prompt = """
Answer the question using only the information provided in the context.

Context:
Employees must inform their manager before working remotely.

Question:
What is the company's maternity leave policy?

Answer:
"""

answer = generate_question_aware_answer(unknown_prompt)

print("\nGenerated Answer for Unknown Question:")
print(answer)