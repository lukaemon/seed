import os
import openai
from dotenv import load_dotenv

davinci = "text-davinci-003"
curie = "text-curie-001"
babbage = "text-babbage-001"
ada = "text-ada-001"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt(prompt, model=curie, temperature=1, max_tokens=100, debug_mode=True):
    try:
        res = openai.Completion.create(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = res.choices[0].text.strip()

        if debug_mode:
            finish_reason = res.choices[0].finish_reason
            print(f"{res.model=}\n{res.usage=}\n{finish_reason=}")
    except Exception as e:
        # Handle other errors
        message = "Error completing text: {}".format(e)

    return message
