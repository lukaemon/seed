## Context
[mm-cot]() involves finetuning T5, but I haven't done it even once before. Since finetuning task specific model is the selling point of huggingface, build some muscle memory couldn't be wrong. 

I know the purpose of gpt-3 is replacing task specific 3b model with few-shot on single LLM, but look at Whisper, Stable Diffusion, Codex, Neeva and sentence-transformers, I want to find an effective symbiosis between specialized small models and LLM. 

## Done

## Learned

## Trigger

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

- Huggingface is default to `AdamW`. For `t5-base`, 250M param model, given k=12 per paper. Expected memory footprint is `(12 + 2 + 2) * 0.25 = 4GB`. How come my peak memory usage is 27G? What's wrong here? 

Try to reduce memory usage:
```python
train_cutoff = 100
ds["train"] = ds["train"].select(range(train_cutoff))
ds["validation"] = ds["validation"].select(range(20))

ds["test"] = ds["test"].select(range(20))
```
- baseline: 27G, 45sec, 175 tflops
- `gradient_accumulation_steps=8`: 27G, 43sec, 174 tflops
- `adafactor`: 27G, 45sec, 175 tflops
- `gradient_checkpointing=True` 16.5, over 11min, Orz
- `fp16`: 27G, 45sec, 175 tflops
- `fp32`: 36G, 47sec, 175 tflops
- `adamw_apex_fused`: 27G, 45sec, 175 tflops
- `model.parallelize()`: 20G, 28 sec, 129 tflops, large still OOM. 
- single GPU: 17g, 68sec, 129 tflops, what the hell? Large OOM. 
- large, baseline bs=1 OOM on 4*a6000. This couldn't be right. 


Nothing works for large. 3 possible solutions:
1. try deepspeed
2. try pytorch FSDP
3. try t5x

- Before I understand the anatomy of memory footprint during training, they are just throwing shits at the wall and see what sticks strategy. 
- [ZeRO paper](http://arxiv.org/abs/1910.02054) helps. Model state + activation + buffer are easy 40x+ multiple with mixed precision fp16 training setup, but in my case, 250m -> 27g is ~100x. What did I miss?
- This is an embarrassing wake up call. I could read PaLM and Sparrow all day but in reality, finetuning t5-large for a hello world level summary task is a big challenge for now. 


