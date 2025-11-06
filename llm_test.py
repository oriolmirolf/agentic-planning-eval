# %% 
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

model = "gpt-4.1-2025-04-14"
prompt = "what model are you?"
base = {"model": model, "input": prompt}

resp = client.responses.create(**{**base})
print(resp.output[0].content[0].text)

# %%
print(resp.output[0].content[0].text)
