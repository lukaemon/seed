from typing import List
from utils import logger

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def load_hf_model(checkpoint: str) -> callable:
    logger.info(f"loading model from {checkpoint}...")

    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def t2t(batch: List[str]):
        inputs = tokenizer(
            batch,
            padding="longest",
            max_length=4096,
            truncation=True,
            return_tensors="pt",
        )

        outputs = model.generate(  # greedy decoding. SC would be used for sampling later.
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=256,
            temperature=0,
            # num_beams=4,
            # early_stopping=True,
        )

        output_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_sequences = [s.strip() for s in output_sequences]

        return output_sequences

    return t2t


def load_oai_model(checkpoint: str) -> callable:
    logger.info(f"loading model from {checkpoint}...")
    pass
