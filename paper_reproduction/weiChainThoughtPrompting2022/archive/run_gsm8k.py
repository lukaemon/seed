import argparse
import re
import os
import json
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from utils import logger
from prompt import math_word_problem_template


TODAY = date.today().strftime("%Y%m%d")
OUTPUT_DIR = f"./output/{TODAY}/gsm8k"


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="google/flan-t5-xxl",
        help="checkpoint for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--dataset_cutoff",
        type=int,
        default=None,
        help="dataset cutoff for evaluation",
    )

    args = parser.parse_args()
    return args


def process_dataset(dataset_cutoff, tokenizer):
    dataset = (
        load_dataset("gsm8k", "main", split="test")
        .map(lambda example: {"label": regex_answer(example["answer"])})
        .map(math_word_problem_template)
        .map(lambda examples: tokenizer(examples["prompt"]), batched=True)
    )
    if dataset_cutoff:
        dataset = dataset.select(range(dataset_cutoff))

    logger.info(
        f"Dataset loaded: {dataset.builder_name}/{dataset.config_name}. Cutoff = {dataset_cutoff}"
    )

    return dataset


def build_dataloader(tokenizer, dataset, batch_size):
    # padding is done in collate_fn, longest per batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=1024,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # dataloader won't accept string columns
    logger.info(f"DataLoader is ready")
    return DataLoader(
        dataset.remove_columns(["question", "answer", "prompt"]),
        collate_fn=data_collator,
        batch_size=batch_size,
    )


def regex_answer(source_text: str) -> int:
    """extract answer from source dataset answer text

    ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
    #### 18'
    """
    match = re.search(r"####\s(\d+)", source_text)
    answer = int(match.group(1)) if match else -1

    return answer


def regex_predict(target_text: str) -> int:
    """extract answer from generated text

    ex: '"She eats 3 + 4 = 7 eggs every day.
    She has 16 - 7 = 9 eggs left.
    She sells 9 * 2 = $18 at the farmers' market.
    The answer is 18."'
    """
    match = re.search(r"The answer is (\d+)", target_text)
    predict = int(match.group(1)) if match else -1
    return predict


def main():
    args = parse_args()
    accelerator = Accelerator()
    logger.info(accelerator.state)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    dataset = process_dataset(args.dataset_cutoff, tokenizer)

    eval_dataloader = build_dataloader(tokenizer, dataset, args.batch_size)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info(f"Loading model {args.checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()
    num_batches = len(eval_dataloader)

    progress_bar = tqdm(range(num_batches))
    progress_bar.set_description(f"Evaluating {args.checkpoint}")

    logger.info(
        f"Evaluating model {args.checkpoint}, {num_batches=}, {args.batch_size=}"
    )

    label = []
    prediction = []
    output_text = []

    for batch in eval_dataloader:
        with torch.no_grad():
            batch_output = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=256,
                temperature=0,
            )
            batch_output_text = tokenizer.batch_decode(
                batch_output, skip_special_tokens=True
            )
            batch_pred = list(map(regex_predict, batch_output_text))

            output_text += accelerator.gather(batch_output_text)
            prediction += accelerator.gather(batch_pred)
            label += accelerator.gather(batch["label"])

            progress_bar.update(1)

    label = np.array(label)
    prediction = np.array(prediction)
    accuracy = (label == prediction).mean()

    # cache failed index
    failed_index = np.where(label != prediction)[0]
    logger.info(f"Failed index cached at failed_index")

    # cache output text
    dataset = dataset.add_column(name="output_text", column=output_text)

    # cache prediction
    dataset = dataset.add_column(name="prediction", column=prediction)
    logger.info(f"otuput_text and prediction cached at dataset")

    result = {
        "checkpoint": args.checkpoint,
        "accuracy": accuracy,
        "total": len(dataset),
        "failure": len(failed_index),
        "regex_parsing_failure": int((prediction == -1).sum()),
    }
    logger.info(f"{result}")

    df = dataset.select(failed_index).to_pandas()

    df = df[["question", "answer", "label", "output_text", "prediction"]]

    # output path
    output_path = f"{OUTPUT_DIR}/{args.checkpoint.split('/')[-1]}"
    os.makedirs(output_path, exist_ok=True)

    # output result as json file
    result_path = f"{output_path}/result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    # output failed cases as json file
    failed_path = f"{output_path}/failed.json"
    with open(failed_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)

    logger.info(f"Result and failed cases saved at {output_path}")


if __name__ == "__main__":
    main()
