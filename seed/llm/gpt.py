import os
import openai
from seed.util import logger


davinci = "text-davinci-003"
curie = "text-curie-001"
babbage = "text-babbage-001"
ada = "text-ada-001"


class GPT:
    def __init__(self, checkpoint=curie):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.checkpoint = checkpoint

    def __call__(self, prompt, temperature=1, top_p=0.8, max_tokens=512):
        """Temp and top_p from Sparrow paper"""
        res = openai.Completion.create(
            prompt=prompt,
            model=self.checkpoint,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = res.choices[0].text.strip()

        # logging
        finish_reason = res.choices[0].finish_reason
        logger.debug(f"\n{self.checkpoint=}\n{res.usage=}\n{finish_reason=}")

        return message

    @property
    def model_name(self):
        return self.checkpoint
