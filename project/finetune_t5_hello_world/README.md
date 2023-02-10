## Context
[mm-cot]() involves finetuning T5, but I haven't done it even once before. Since finetuning task specific model is the selling point of huggingface, build some muscle memory couldn't be wrong. 

I know the purpose of gpt-3 is replacing task specific 3b model with few-shot on single LLM, but look at Whisper, Stable Diffusion, Codex, Neeva and sentence-transformers, I want to find an effective symbiosis between specialized small models and LLM. 

## Done
- 3 hello world summary finetuning on t5 series. 

## Learned
- General API of huggingface `trainer`. 
- The quest to fight OOM leads me to init exploration of training optimization and details of padding/truncation. 
- The nature of relational position embedding and the artificial model max_length imposed on the t5 series. 
- Do simple data munging to understand the shape of data would prevent many stupid mistakes later. 
  - Eye ball few examples. 
  - Do the task by yourself to get a sense of what you are going to ask the model to do.
  - Simple statistics on the dataset, such as token length distribution, token frequency, etc.

## Trigger
- Push training hardware utilization
  - [Change vocabulary size for better throughput](https://twitter.com/karpathy/status/1621578354024677377). 
    - [How can I change vocab size for pretrained model?](https://github.com/huggingface/transformers/issues/237)
  - Learn to use accelerate, deepspeed, FSDP, composer, collossal, t5x. 
- Inference
  - Start from huggingface API
  - Follow the [Neeva route](https://twitter.com/neeva/status/1622640441064579076?s=12&t=MjCpOKlzFcDn81EVM_HUFg)
  - Read [Speculative Sampling](https://arxiv.org/abs/2302.01318) and [Scaling Transformer Inference](https://arxiv.org/abs/2211.05102). 
- Better understanding of finetuning to adopt to different tasks
  - Read [The Flan Collection](http://arxiv.org/abs/2301.13688)
- Build t5 model from scratch, control every single bit of the model and training. This would be necessary stepping stone to multimodal research since lots of glue architecture requires model surgery. mm-cot has very light model modification for `gated fusion`, great init for me. 


## Log
### Out of memory
#### My training config baseline:
```python
args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=2,
    bf16=True,  # TODO: figure out bf16 vs fp16
    gradient_accumulation_steps=1,
    predict_with_generate=True,
    hub_model_id=hub_model_id,
    report_to="wandb",
)
```
- Easily out of memory on my 2*3090 machine. Settle with batch size 1. Even with that, t5-large still OOM.
- Strongly suspect I miss something important. Light years away from the optimal training wrt computation resource utilization. 

#### mm-cot's training config:
- `mm-cot`'s training config is very much the same as mine above. It didn't even use `bf16`. However, their experiments were ran on `4 NVIDIA Tesla V100 32G`... Maybe I was just naive to think I could do the same on `t5-large` with 2*3090. But **how to compute the memory requirement of finetuning a model?**

#### Dealing with OOM
In this [ZeRO blog](https://www.microsoft.com/en-us/research/blog/ZeRO-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
![](https://www.microsoft.com/en-us/research/uploads/prod/2020/02/DeepSpeed-Image-1.png)
>  As a specific example, we show the memory consumption for a 7.5B parameter model using Adam optimizer where **K=12** on 64 GPUs. We also show the communication volume of ZeRO relative to the baseline.

[Try to reduce memory usage](https://huggingface.co/docs/transformers/v4.18.0/en/performance):
```python
tcut = 100
vcut = 20
ds["train"] = ds["train"].select(range(tcut))
ds["validation"] = ds["validation"].select(range(vcut))

ds["test"] = ds["test"].select(range(vcut))
```
- baseline: 27G, 45sec, 175 tflops
- `gradient_accumulation_steps=8`: 27G, 43sec, 174 tflops
- `adafactor`: 27G, 45sec, 175 tflops
- `gradient_checkpointing=True` 16.5, over 11min, Orz
- `fp16`: 27G, 45sec, 175 tflops
- `fp32`: 36G, 47sec, 175 tflops
- `adamw_apex_fused`: 27G, 45sec, 175 tflops
- `model.parallelize()`: 20G, 28 sec, 129 tflops, GPU utilization drops to ~50. Large OOM. 
- single GPU: 17g, 68sec, 129 tflops, what the hell? Large OOM. 
- baseline bs=1 on 4*a6000. Large OOM. This couldn't be right. 


Nothing works. 3 possible solutions:
1. try deepspeed
2. try pytorch FSDP
3. try t5x

- Before I understand the anatomy of memory footprint during training, they are just throwing shits at the wall and see what sticks strategy. 
- [ZeRO paper](http://arxiv.org/abs/1910.02054) helps. Model state + activation + buffer are easy 40x+ multiple with mixed precision fp16 training setup, 40x with 250m model = 10g max. but in my case, 250m -> 27g is ~100x. What did I miss?
- This is an embarrassing wake up call. I could read PaLM and Sparrow all day but in reality, finetuning t5-large for a hello world level summary task is a big challenge for now. 
- [mm-cot source code](https://github.com/amazon-science/mm-cot/blob/main/main.py#L32) cuts input_len at 512. My `max_length=4096` right now. Would that be the problem? Bingo!
  - `max_length=512` + `model.parallelize()`: 6g, 25sec, 91 tflops...
  - `max_length=512`: 7g, 43sec, 110 tflops
- I thought about model param, optimizer, but totally forget context window and the exponential memory nature of input length. 

#### T5 max length
```
/usr/local/lib/python3.8/dist-packages/transformers/models/t5/tokenization_t5_fast.py:155: FutureWarning: This tokenizer was incorrectly instantiated with a model max length of 512 which will be corrected in Transformers v5.
For now, this behavior is kept to avoid breaking backwards compatibility when padding/encoding with `truncation is True`.
- Be aware that you SHOULD NOT rely on t5-base automatically truncating your input to 512 when padding/encoding.
- If you want to encode/pad to sequences longer than 512 you can either instantiate this tokenizer with `model_max_length` or pass `max_length` when encoding/padding.
```
- [Warning tells you you will get indexing errors in T5 for going beyond max length](https://github.com/huggingface/transformers/issues/16986#issuecomment-1112190230)
- [What is the maximum input seq length, when using T5-3b model?](https://github.com/google-research/text-to-text-transfer-transformer/issues/273)
- [T5 Model : What is maximum sequence length that can be used with pretrained T5 (3b model) checkpoint?](https://github.com/huggingface/transformers/issues/5204)
- [[T5 Tokenizer] Model has no fixed position ids - there is no hardcode](https://github.com/huggingface/transformers/pull/16990)

[Padding and truncation
](https://huggingface.co/docs/transformers/main/en/pad_truncation#padding-and-truncation): this setting means **padding to max sequence in batch**. 
```python
model_input = tokenizer(
    inputs,
    padding=True,
    pad_to_multiple_of=8,
    truncation=True,
)
```
- By default T5 should not have a set maximum length. 
- The setting works on Large, bs=8, XL is still OOM.
- I like this set up more. All tokenizer settings are in one place. Keep collator args clean.
- `m.parallelize()` has lower GPU utilization but faster runtime and more even GPU memory allocation. Why?
