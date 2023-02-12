## Done
- Build [BBH huggingface dataset](https://huggingface.co/datasets/lukaemon/bbh). Eval BBH across all `flan-t5` familiy. 
- Build [MMLU huggingface dataset](https://huggingface.co/datasets/lukaemon/mmlu). Eval MMLU across all `flan-t5` familiy.

## Learned
### About engineering
- HF dataset is a good way to organize data. A must for repdoducible research.
- BBH test time is `770min` for all t5 model. GPU usage is ~50% for each. Naive `model.parallel()` is not effective. MMLU took even longer. I have to dig in hf transformers or switch to t5x for better computation utilization.
- Eval LLM the way people would use it. Stick to `text2text`. 

### About CoT
- **Performance gains from CoT prompting only emerge with sufficient model size**.  
- T5 can't do CoT, even with xxl. Flan, as finetuninig, can't help t5 do CoT.  
- **instruction finetuning improves unseen tasks when the unseen tasks are in the same prompting paradigm as the finetuning tasks**.
  - Flan improves testing performance because the instruction finetuning helps t5 get familiar with the prompt distribution of exam, multiple choice task. These tests usually instruct the testee to do something. Calculate this, find that, etc. 
- Flan includes few CoT tasks, which helps LLM to keep the CoT capability, but that is not enough to make t5 do CoT. I don't think you can ft small model to do CoT. As learned from CoT paper, small model makes mistakes here and there, generate low quality CoT chain, which usually leads to wrong conclusion. 
- (henighanScalingLawsAutoregressive2020): significant semantic information may lie in the **last few bits**. To get to those bits, scale is necessary to get semantics right. 
- T5 is not the right model for `rationale engineering` research. I should use `dacinci`. Maybe use `UL2` as local davinci for the poor. 
- Even though CoT didn't work on T5, SC works, gsm8k `16.7%` -> `~20%`. They are different methods to scale computation on sovling tasks. CoT's linear chain make it hard for small model to get right. However, voting seems to cancel out mistakes and improve the performance in aggregate, at huge cost of computation.
  - SC is not cost effective way of using flops with T5.
  - Can't vote on open ended tasks, such as summary. Recitation requires extensive world knowledge, which doesn't exist in small model.

### About finetuninig
#### For LLM 
- Finetuning is mostly for UX improvement. Better be general enough to benefit all tasks. With the right prompt, few shot, and limited interaction mode (ex: multichoice), raw pretrained model could perform as well as finetuned model. However, prompt engineering is not good UX. For open ended tasks, prompt engineering and few shot is not feasible at all (ex: ChatGPT, MedPaLM). 
- Finetuninig won't bring out new capability. Just enhance existing ones in pretrained model. If base model is too weak, it's beyond tuning, ex: [OPT-IML](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML).
- Need new objective function to pretrain LLM for new capability. `UL2` and `FIM` are good examples. `UL2R` is cost effective way to mold existing autoregressive LLM.
- Be very careful to choose what to finetune, at what cost? Cost wrt what general capability is overriden due to this finetuning session?
- Leave the task apdaptation to `PEFT` and `prompt engineering`. Keep finetuning general. ex: MedPaLM has no large scale finetuninig after Flan, instead, opts for **instruction prompt tuning** to adapt to medical use case. PEFT is the right abstraction layer to do task adaptation for LLM. 

#### For T5
- Choose target task strategically and laser focus on one task per model. Finetune the hell out of it to get the best performance. You may not need 11b for every task. Keep it as lean as possible. 
- `PEFT` and `prompt engineering` is not the right strategy for T5. T5 is all in mode. Don't try to build a 11b model that beats 540b right now. Follow the trend for a while. 

### About building application
- LLM API would do 80% of the work. It's universal glue. The rest would be few highly optimized T5 and other tools such as vector database, etc.
- For super valuable, large TAM problem, tight vertical intergration is good startegy, like Neeva, which is capital and talent intensive. Recognize what can be done and what's beyond means right now, accept, act and iterate as fast as possible. 
- Let me get the 80% right first. `rationale engineering`, `in-context learning`, best practice of `prompt engineering`, etc. Then specialize on finetuning t5 models. 
- Cultivate good taste on what is strategically valueable task to build customized model for. 

### New questions
- Why codex 12b works so well? 
  - Coding is such a limisted way of manipulating symbols that could be saturated by small model?
  - Can we saturate math with 12b as well?
  - Does that mean it's possible to saturate formal reasoning with 12b? 
- How about legal application? 

## Log
### Working on MMLU
- Same setup as BBH, except for using 5 shot.
- Source code from `MMLU` author repo for his approach to multi choice evaluation.
```python
logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()

probs = (
    torch.nn.functional.softmax(
        torch.tensor(
            [
                logits[tokenizer("A").input_ids[0]],
                logits[tokenizer("B").input_ids[0]],
                logits[tokenizer("C").input_ids[0]],
                logits[tokenizer("D").input_ids[0]],
            ]
        ),
        dim=0,
    )
    .detach()
    .cpu()
    .numpy()
)
pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
```
- This is crazy... My approach is close to google's. Stick to text2text. Leave probability alone. When eval human intelligence, one won't ask for monitoring neuronal firing patterns as proxy to the quality of response. 
- xxl with CoT oom even with batch_size=1. Orz. 

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