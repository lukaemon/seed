import logging
import openai
from config.constant import OPENAI_API_KEY

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

    # logging
    finish_reason = res.choices[0].finish_reason
    logging.debug(f"{res.model=}\n{res.usage=}\n{finish_reason=}")

    return message
