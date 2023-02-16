## tl;dr

## Context
- Finetuning is expensive on both computation resource and dataset curation. PEFT seems to be the logical next step to reduce the cost of task adaptation. 
- Huggingface just released a PEFT library. 
- As a preparation to BLIP2 project. I see BLIP2 as advanced, multimodal prompt tuning. Better get familiar with the concept and the tooling asap.
- Still feel the burning realization that I can't finetune a 1b+ model locally with 2*3090. Not to compete with TPU pods 😅 but consumer level hardware could definitely do better. PEFT is a step in the right direction.

## Done

## Learned

## Next?

## Log
- Read the paper, [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691). Start with all hello world examples from peft lib and get it running first. 
- `prompt tuning` style `PEFT` is a perfect fit to `load_in_8bit`.  
  - Whole LM is frozen. Only need inference level of memory for model and small amount for prompt tuning.
  - It is significant because full model finetuning is easy 30x+ model size memory footprint. To finetune an `opt-6.7b` would require 200+ GB of cuda memory even with batch_size=1. 
  - Prompt tuning make it possible to finetune a 6.7B model on a Nvidia 3080 12g with batch_size > 1. 
  - 💡Very enabling technology. Combine  `int8` inference, `PEFT`, `fp8` training, consumer level hardware could actually do a lot. 
  - With a bit more computation budget, adaptor style `PEFT` may yield even better result.
  - [Networking would be a problem for fp8 training](https://twitter.com/Tim_Dettmers/status/1621930955673047040). That's for distributed LLM training. Making training and inference a T5-xxl, 11b, locally with consumer level hardware is a big deal and no worry about networking in that case. Modern consumer hardware could easily run majority of open source software, but can't finetune a 1b+ model.
- `apex` doesn't play well with `int8` inference. How to turn it off?
  ```
  │ /usr/local/lib/python3.8/dist-packages/apex/normalization/fused_layer_norm.py:69 in forward      │
  │                                                                                                  │
  │    66 │   │   ctx.eps = eps                                                                      │
  │    67 │   │   input_ = input.contiguous()                                                        │
  │    68 │   │   weight_ = weight.contiguous()                                                      │
  │ ❱  69 │   │   output, invvar = fused_layer_norm_cuda.rms_forward_affine(                         │
  │    70 │   │   │   input_, ctx.normalized_shape, weight_, ctx.eps)                                │
  │    71 │   │   ctx.save_for_backward(input_, weight_, invvar)                                     │
  │    72 │   │   return output                                                                      │
  ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
  RuntimeError: expected scalar type Float but found Half
  ```
-