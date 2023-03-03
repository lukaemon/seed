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
from tqdm.auto import tqdm

from utils import logger
from prompt import math_word_problem_template, ul2_preprocess


OUTPUT_DIR = f"./output/gsm8k"


class GSM8K:
    def __init__(self, checkpoint, batch_size=16, dataset_cutoff=None):
        self.batch_size = batch_size
        self.dataset_cutoff = dataset_cutoff

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.dataset = None  # ['question', 'answer', 'label', 'prompt', 'input_ids', 'attention_mask', 'output_text', 'prediction']
        self.result = None
        self.failed_index = None

    def load_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")
        if self.dataset_cutoff:
            self.dataset = self.dataset.select(range(self.dataset_cutoff))

        logger.info(
            f"Dataset loaded: {self.dataset.builder_name}/{self.dataset.config_name}. Cutoff = {self.dataset_cutoff}"
        )

    def process_dataset(self):
        logger.info(f"Processing dataset")

        self.dataset = self.dataset.map(
            lambda example: {"label": GSM8K.regex_answer(example["answer"])}
        ).map(math_word_problem_template)

        # ul2 [S2S] prompt makes the model performance worse.
        # if self.checkpoint == "google/ul2":
        #     self.dataset = self.dataset.map(
        #         lambda example: {"prompt": ul2_preprocess(example["prompt"])}
        #     )
        #     logger.info("Process prompt for UL2")

        # tokenization
        self.dataset = self.dataset.map(
            lambda examples: self.tokenizer(examples["prompt"]), batched=True
        )

    def build_dataloader(self):
        self.load_dataset()
        self.process_dataset()

        # padding is done in collate_fn, longest per batch
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=1024,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        # dataloader won't accept string columns
        logger.info(f"DataLoader is ready")
        return DataLoader(
            self.dataset.remove_columns(["question", "answer", "prompt"]),
            collate_fn=data_collator,
            batch_size=self.batch_size,
        )

    def eval(self):
        eval_dataloader = self.build_dataloader()

        logger.info(f"Loading model {self.checkpoint}")
        model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        )
        model.parallelize()
        num_batches = len(eval_dataloader)

        progress_bar = tqdm(range(num_batches))
        progress_bar.set_description(f"Evaluating {self.checkpoint}")

        logger.info(
            f"Evaluating model {self.checkpoint}, {num_batches=}, {self.batch_size=}"
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
                batch_output_text = self.tokenizer.batch_decode(
                    batch_output, skip_special_tokens=True
                )
                batch_pred = list(map(GSM8K.regex_predict, batch_output_text))

                output_text += batch_output_text
                prediction += batch_pred
                label += batch["label"]

                progress_bar.update(1)

        del model
        torch.cuda.empty_cache()
        logger.info("Release model and cuda cache")

        label = np.array(label)
        prediction = np.array(prediction)
        accuracy = (label == prediction).mean()

        # cache failed index
        self.failed_index = np.where(label != prediction)[0]
        logger.info(f"Failed index cached at self.failed_index")

        # cache output text
        self.dataset = self.dataset.add_column(name="output_text", column=output_text)

        # cache prediction
        self.dataset = self.dataset.add_column(name="prediction", column=prediction)
        logger.info(f"otuput_text and prediction cached at self.dataset")

        self.result = {
            "checkpoint": self.checkpoint,
            "accuracy": accuracy,
            "total": len(self.dataset),
            "failure": len(self.failed_index),
            "regex_parsing_failure": int((prediction == -1).sum()),
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

        logger.info(f"Result and failed cases saved at {output_path}")

    @staticmethod
    def regex_answer(source_text: str) -> int:
        """extract answer from source dataset answer text

        ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
        She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
        #### 18'
        """
        match = re.search(r"####\s(\d+)", source_text)
        answer = int(match.group(1)) if match else -1

        return answer

    @staticmethod
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
