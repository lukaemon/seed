import os

import openai
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_RESPONSE_TOKENS = 128


def dry_run_price_estimation(tokenizer, dataset, k):
    """
    tokenizer: gpt2 tokenizer
    dataset: hf dataset
    k: number of samples to generate per each input
    """
    token_cnt = 0

    tokens = tokenizer(dataset["prompt"])
    for t in tokens["input_ids"]:
        token_cnt += len(t) + MAX_RESPONSE_TOKENS * k

    davinci = 0.02  # per k
    curie = 0.002  # per k

    print(
        f"total tokens upper bound: {token_cnt}\ndavinci price: {token_cnt / 1000 * davinci:.2f}\ncurie price: {token_cnt / 1000 * curie:.2f}"
    )


def complete(model, prompt, temp, k):
    """for one prompt"""
    res = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temp,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=k,
    )

    texts = [r["text"] for r in res["choices"]]

    return texts
