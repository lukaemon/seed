# Natively, T5 series are useless as convo agent backend.
# You need serious finetuning to get 11b model to work for the task.

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

flan_t5_xl = "google/flan-t5-xl"  # 3b

# bf16, GPU ram 21g. CPU ram 35g
flan_t5_xxl = "google/flan-t5-xxl"  # 11b

t03b = "bigscience/T0_3B"  # 3b

# bf16, GPU ram 21g. CPU ram 40g
# with low_cpu_mem_usage=True, CPU ram 66g -> 40g, loading time 1m40s -> 48s
t0pp = "bigscience/T0pp"  # 11b
t0 = "bigscience/T0"  # 11b

t5_lm = "google/t5-xxl-lm-adapt"  # 11b

# bf16, GPU ram 40g. CPU ram 40g
# with low_cpu_mem_usage=True, CPU ram 80g -> 40g, loading time 2m32s -> 19s
ul2 = "google/ul2"  # 20b


class T5:
    def __init__(self, checkpoint=flan_t5_xxl):
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
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
