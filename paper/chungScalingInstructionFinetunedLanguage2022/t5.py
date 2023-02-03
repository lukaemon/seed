import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)


def load_model(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()

    return model, tokenizer


def predict(model, tokenizer, batch, temp=0, k=1):
    """
    batch: dict with keys "input_ids" and "attention_mask", whic should be the output of collator
    k: number of samples to generate per each input

    returns: [batch_size, k], list of lists of strings
    """

    batch_output = model.generate(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        max_length=256,
        do_sample=temp > 0,  # won't sample without this
        temperature=temp,
        num_return_sequences=k,
    )

    batch_output_seq = tokenizer.batch_decode(batch_output, skip_special_tokens=True)

    if k > 1:
        return [batch_output_seq[i : i + k] for i in range(0, len(batch_output_seq), k)]
    else:
        return batch_output_seq
