from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

mt0_large = "bigscience/mt0-large"  # 1.2b
mt0_xl = "bigscience/mt0-xl"  # 3.7b
mt0_xxl = "bigscience/mt0-xxl"  # 13b


class MT0:
    def __init__(self, checkpoint=mt0_xxl):
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, device_map="auto", load_in_8bit=True
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
