# Natively, both are useless as convo agent backend.
# That's understandable. You need serious finetuning to get 11b model to work for the task.

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

flan_t5_xl = "google/flan-t5-xl"  # 3b

# bf16 works, GPU ram settled around 21g total. CPU ram peaked around 35g
flan_t5_xxl = "google/flan-t5-xxl"  # 11b

t03b = "bigscience/T0_3B"  # 3b

# bf16 works, GPU ram settled around 21g total. CPU ram peaked around 65.8g
t0pp = "bigscience/T0pp"  # 11b


class T5:
    def __init__(self, checkpoint=flan_t5_xxl):
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
        )
        self.model.parallelize()

    def __call__(self, prompt, temperature=1, top_p=0.8, max_tokens=512):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs, max_length=max_tokens, temperature=temperature, top_p=top_p
        )
        message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return message

    @property
    def model_name(self):
        return self.checkpoint
