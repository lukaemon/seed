import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINFACEHUB_API_TOKEN = os.getenv("HUGGINFACEHUB_API_TOKEN")

openai.api_key = OPENAI_API_KEY
davinci = "text-davinci-003"
curie = "text-curie-001"
babbage = "text-babbage-001"
ada = "text-ada-001"


# Temp and top_p from Sparrow paper
def gpt(prompt, temperature=1, top_p=0.8, max_tokens=512):
    res = openai.Completion.create(
        prompt=prompt,
        model=curie,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    message = res.choices[0].text.strip()
    return message
