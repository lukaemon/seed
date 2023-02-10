import os
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
from datasets import load_dataset
import evaluate
import nltk
import numpy as np
import wandb
import fire

nltk.download("punkt", quiet=True)
wandb.init(project="finetune-t5-hello-world")


def main(
    dataset_name,
    gpu,
    lr=3e-4,
    bs=4,
    epochs=1,
    max_length=1024,
    padding=True,
    pad_to_multiple_of=8,
    checkpoint="google/flan-t5-base",
):
    gpu = str(gpu)
    if gpu in ["0", "1"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # visible GPU

    utc_time_in_seconds = int(time.time())
    ft_output_dir = os.getenv("HF_FINETUNE_OUTPUT_DIR")
    model_name = checkpoint.split("/")[-1]
    hub_model_id = f"{model_name}-{dataset_name}-{utc_time_in_seconds}"
    model_output_dir = os.path.join(ft_output_dir, hub_model_id)

    wandb.config.dataset_name = dataset_name
    lr = float(lr)

    ds = load_dataset(dataset_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    if gpu == "all":
        model.parallelize()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_column = "dialogue" if dataset_name == "samsum" else "document"

    def preprocess(examples):
        output = tokenizer(
            examples[input_column], max_length=max_length, truncation=True
        )
        output["labels"] = tokenizer(examples["summary"])["input_ids"]
        return output

    tk_ds = ds.map(preprocess, batched=True).remove_columns(ds["train"].column_names)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return result

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=padding, pad_to_multiple_of=pad_to_multiple_of
    )

    args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        optim="adafactor",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=epochs,
        bf16=True,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        hub_model_id=hub_model_id,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tk_ds["train"],
        eval_dataset=tk_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
