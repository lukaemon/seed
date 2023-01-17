import re
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)
from tqdm.auto import tqdm

from utils import logger
from prompt import math_word_problem_template


OUTPUT_DIR = "./output/gsm8k"


class GSM8K:
    def __init__(self, checkpoint, batch_size=8, dataset_cutoff=None):
        self.batch_size = batch_size
        self.dataset_cutoff = dataset_cutoff
        self.checkpoint = checkpoint

        self.dataset = None  # ['question', 'answer', 'label', 'prompt', 'input_ids', 'attention_mask', 'output_text', 'prediction']
        self.result = None
        self.failed_index = None  # failed index

    def build_dataloader(self, tokenizer):
        logger.info(f"building dataloader...")
        self.dataset = (
            load_dataset("gsm8k", "main", split="test")
            .map(lambda example: GSM8K.regex_answer(example["answer"]))
            .map(math_word_problem_template)
            .map(lambda examples: tokenizer(examples["prompt"]), batched=True)
        )

        if self.dataset_cutoff:
            logger.info(f"dataset cutoff = {self.dataset_cutoff}")
            self.dataset = self.dataset.select(range(self.dataset_cutoff))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            max_length=1024,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        # dataloader won't accept string columns
        return DataLoader(
            self.dataset.remove_columns(["question", "answer", "prompt"]),
            collate_fn=data_collator,
            batch_size=self.batch_size,
        )

    def eval(self, checkpoint):
        self.checkpoint = checkpoint

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        eval_dataloader = self.build_dataloader(tokenizer)

        logger.info(f"loading model {checkpoint}...")
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        )
        model.parallelize()
        num_batches = len(eval_dataloader)

        progress_bar = tqdm(range(num_batches))
        progress_bar.set_description(f"evaluating {checkpoint}")

        logger.info(
            f"evaluating model {checkpoint}, {num_batches=}, {self.batch_size=}"
        )

        label = []
        prediction = []
        output_text = []

        for batch in eval_dataloader:
            with torch.no_grad():
                batch_output = model.generate(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    max_length=256,
                    temperature=0,
                )
                batch_output_text = tokenizer.batch_decode(
                    batch_output, skip_special_tokens=True
                )
                batch_pred = list(map(GSM8K.regex_predict, batch_output_text))

                output_text += batch_output_text
                prediction += batch_pred
                label += batch["label"]

                progress_bar.update(1)

        del model
        torch.cuda.empty_cache()
        logger.info("release model and cuda cache")

        label = np.array(label)
        prediction = np.array(prediction)
        accuracy = (label == prediction).mean()

        # cache failed index
        self.failed_index = np.where(label != prediction)[0]
        logger.info(f"failed index cached at self.failed_index")

        # cache output text
        self.dataset = self.dataset.add_column(name="output_text", column=output_text)

        # cache prediction
        self.dataset = self.dataset.add_column(name="prediction", column=prediction)
        logger.info(f"otuput_text and prediction cached at self.dataset")

        self.result = {
            "checkpoint": checkpoint,
            "accuracy": accuracy,
            "n_total": len(self.dataset),
            "n_failed": len(self.failed_index),
            "regex_error": int((prediction == -1).sum()),
        }
        logger.info(f"{self.result}")

    def generate_report(self, max_fail_case=None):
        df = self.dataset.select(self.failed_index).to_pandas()
        if max_fail_case:
            df = df.head(max_fail_case)

        df = df[["question", "answer", "label", "output_text", "prediction"]]

        # output path
        output_path = f"{OUTPUT_DIR}/{self.checkpoint.split('/')[-1]}"
        os.makedirs(output_path, exist_ok=True)

        # output result as json file
        result_path = f"{output_path}/result.json"
        with open(result_path, "w") as f:
            json.dump(self.result, f, indent=4)

        # output failed cases as json file
        failed_path = f"{output_path}/failed.json"
        with open(failed_path, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=4)

        logger.info(f"result and failed cases saved at {output_path}")

    @staticmethod
    def regex_answer(source_text):
        """extract answer from source dataset answer text

        ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
        She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
        #### 18'
        """
        match = re.search(r"####\s(\d+)", source_text)
        answer = int(match.group(1)) if match else -1

        return {"label": answer}

    @staticmethod
    def regex_predict(target_text):
        """extract answer from generated text

        ex: '"She eats 3 + 4 = 7 eggs every day.
        She has 16 - 7 = 9 eggs left.
        She sells 9 * 2 = $18 at the farmers' market.
        The answer is 18."'
        """
        match = re.search(r"The answer is (\d+)", target_text)
        predict = int(match.group(1)) if match else -1
        return predict
