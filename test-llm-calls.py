# %% 
from openai import OpenAI

client = OpenAI()

model = "gpt-4.1-2025-04-14"
prompt = "Hello world"
base = {"model": model, "input": prompt}

resp = client.responses.create(**{**base})
print(resp.text)

# %%