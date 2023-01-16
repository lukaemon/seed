import logging
import re

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger("CoT")
logger.setLevel(logging.INFO)


def build_t2t(checkpoint: str) -> callable:
    logger.info(f"loading model from {checkpoint}...")

    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def t2t(batch):
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


def regex_target(source_text):
    """extract answer from dataset source text
    ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
    #### 18'
    """
    match = re.search(r"####\s(\d+)", source_text)

    return int(match.group(1)) if match else None


def regex_predict(target_text):
    """extract answer from generated text
    ex: '"She eats 3 + 4 = 7 eggs every day.
    She has 16 - 7 = 9 eggs left.
    She sells 9 * 2 = $18 at the farmers' market.
    The answer is 18."'
    """
    match = re.search(r"The answer is (\d+)", target_text)

    return int(match.group(1)) if match else None
