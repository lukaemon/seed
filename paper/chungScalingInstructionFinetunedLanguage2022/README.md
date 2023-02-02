## What I've done?
- Build [BBH huggingface dataset](https://huggingface.co/datasets/lukaemon/bbh).
- Eval BBH with full set of `flan-t5` familiy. 
- Build [MMLU huggingface dataset](https://huggingface.co/datasets/lukaemon/mmlu).

## What I've learned?
- The purpose of finetuning and be very careful to what to finetune. Leave the rest to `PEFT` and prompt engineering. 

## Log
### Working on MMLU
- Same setup as BBH, except for using 5 shot.


### Working on BBH
- Experiment setup:
  - 3 shot BBH. 
  - Including both task instruction and answer options. 
  - Greedy decoding, t=0. 
  - Exact match. 
    - "We extract the final answer based on keywords that the language model is expected to produce (i.e., “the answer is”)."
  - Normalized average per model. 
    - Follow BIG-bench's normalized preferred metric protocol. 
    - "Our normalized average metric is the macro-average over six normalized scores: MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT."
- Token number analysis per prompted instance of the whole BBH
  - cot=False: mean=465.29, min=84, max=1643
  - cot=True: mean=945.07, min=241, max=1971
  - That's why the paper assumes avg input token len = 1.5k. They do both direct prompt and CoT.
  - You don't want prompt cutoff. Max length is set to 2048 for now. Output max length set to 512 so the output CoT and answers won't be cut off.
- `batch_size=8` would trigger cuda oom with xxl model at task 18, `ruin_names`. Huge mem allocation happened. 
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 3.53 GiB (GPU 0; 23.69 GiB total capacity; 18.09 GiB 
already allocated; 3.51 GiB free; 19.86 GiB reserved in total by PyTorch
```

### Reading Flan-T5 paper
- `gsm8k` is in the training set of `flan-t5`, not the test set of course. 
- "most of the gains from multi-task instruction finetuning come from the model learning to better express knowledge that it already knows from pretraining, and more than 282 tasks does not help too much."
  - `UL2` or `FIM` are possible to yield brand new LLM capability. Finetuning is more akin to UX improvement. 
  - Ex: `Flan-PaLM` could do 0-shot `CoT`. However, few shot `CoT` works on native `PaLM` already. Inst ft enables LLM to express CoT more easily, effectively. And LLM users generally would reward CoT as good LLM property. 
  - You should expect all general good properties that could be triggered with few-shot or special prompt to become 0 shot within next few LLM iterations. 
- "it is critical to finetune on some CoT examples in order to keep such reasoning abilities, because finetuning on only non-CoT degrades performance on CoT by a substantial amount"
  - Bigger model may have more room to be finetuned by even 540b would face trade off here. That's why google went both `PEFT` and `UL2` direction.
- Should learn from its eval process for open-ended generation. 
  - temp=0.7
  - rank by log prob without length normalization.
  - removing any generations with scores that were better than half of the median score, which is likely to have undesirable repetitions and so on.
  - present to humans.
- `BBH` uses regex to extract prediction for multiple choice. I did the same.
- `MMLU` -> "For few-shot evaluation, we add up to 5 demonstration examples with answers to the prompt before appending the question. All prompts end with “Answer: ”. The model then produces probabilities for the tokens “A,” “B,” “C,” and “D,” and we treat the highest probability option as the prediction."
  - This is very very far from what LLM would be used in real use case. I prefer pure text2text approach.