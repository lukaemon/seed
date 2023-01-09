from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

t03b = "bigscience/T0_3B"  # 3b
t0pp = "bigscience/T0pp"  # 11b


class T0:
    def __init__(self, checkpoint=t03b):
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, device_map="sequential", torch_dtype=torch.bfloat16
        )

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
