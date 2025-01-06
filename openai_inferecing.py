from openai import OpenAI

client = OpenAI(api_key="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",base_url="http://localhost:8000/v1")

completion = client.chat.completions.create(
    model = "HuggingFaceTB/SmolLM-135M-Instruct",
    messages=[
        {"role": "user", "content": "Who is bill gates?" },
    ]
)

print(completion.choices[0].message.content)