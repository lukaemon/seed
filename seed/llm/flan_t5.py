# https://github.com/huggingface/transformers/issues/20287#issuecomment-1342219429
# https://github.com/huggingface/transformers/pull/20683

from transformers import T5Tokenizer, T5ForConditionalGeneration

flan_t5_xl = "google/flan-t5-xl"
flan_t5_xxl = "google/flan-t5-xxl"


class FlanT5:
    def __init__(self, checkpoint=flan_t5_xxl):
        self.checkpoint = checkpoint

        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, device_map="auto"
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
