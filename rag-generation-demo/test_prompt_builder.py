from prompt_builder import build_rag_prompt


context = """Remote employees can work from home up to three days per week.
Employees must inform their manager before working remotely."""

question = "How many days can employees work from home?"

prompt = build_rag_prompt(context, question)

print(prompt)