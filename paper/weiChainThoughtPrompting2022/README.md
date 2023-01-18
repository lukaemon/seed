## What I've done?
- Reproduce `GSM8k` and `CSQA` eval on t5 series. 
  - `GSM8k`/test: 16.7%, `flan-t5-xxl`
  - `CSQA`/validation: 84.6%, `flan-t5-xxl`. This is better than `PaLM 540b`. Something wrong but I can't find answer of test split. Revisit later. 

## What I've learned?
- `flan-t5-xxl` is the only good open model to play with locally. Use `davinci` to taste cutting edge. 
- Getting good at handling `hf dataset`, `tokenizer`, `pt DataLoader`, logging and prompt. Need those skills and more for future finetuning sessions. 
- `BIG-bench` and `BBH` would be great init step into `t5x` ecosystem. Then I'm ready to dive into finetuning. Compare and contrast the ergonomics and efficiency for ft with hf vs t5x, GPU vs TPU. 
- My hunch is training with TPU, inference with native PyTorch, demo, PR with huggingface. 

## Log
### 20230118 `CSQA`
- `Class GSM8K` is unnecessary abstraction. I do `CSQA` in simple collection of functions which is more straight forward and simpler. 
- `flan-t5-xxl` scores 84.6% on csqa validation split.

### fail case
```
Q: If you are prone to postpone work what will you have to do in order to finish on time?
Answer Choices: 
(a) eat
(b) hasten
(c) antedate
(d) bring forward
(e) advance
A: If you are prone to postpone work, you will have to advance it. So the answer is (e).
Right answer: (b)
```
I think e is ok in this case lol. 

```
Q: She was always helping at the senior center, it brought her what?
Answer Choices: 
(a) satisfaction
(b) heart
(c) feel better
(d) pay
(e) happiness
A: Helping others brings satisfaction. Helping at the senior center brings satisfaction. So the answer is (a).
Right answer: (e)
```
e is definitely ok. 

```
Q: August needed  money because he was afraid that he'd be kicked out of his house.  What did he need money to do?
Answer Choices: 
(a) control people
(b) pay bills
(c) hurt people
(d) buy food
(e) get things
A: August needed money to pay bills. The answer is (b).
Right answer: (b)
```
`regex` parsing bug...

```
Q: What must someone do before they shop?
Answer Choices: 
(a) get money
(b) have money
(c) bring cash
(d) go to market
(e) bring cash
A: Before they shop, someone must have money. Of the above choices, only have money is a verb. So the answer is (b).
Right answer: (a)
```
Again, not LLM's fault. Human language is vague and multiple choice is not the right format to test modern LLM. 

### 20230117 `GSM8K`
- I don't see the value of using `evaluate` library for simple `accuracy` metrics. Maybe later would see the value.
- `ul2` `[S2S]` prompt makes the model performance worse. Why?
- `ul2` output is verbose, repetitive and illogical. Don't know how to use it well. Ignore this model for now. I won't be tweaking small autoregressive model anyway.
- `T0pp` is unable to do CoT, at all.
- `ul2` acc=4%, close to paper result 4.4. `t0pp` ~= 0. So locally, `flan-t5`, 16.7% is the only open model worth playing wrt CoT. It's at the same level as `LaMDA` 137b, 14.3%. 
- Adopt to HF `accelerate` didn't improve thoughput. Current ipynb setup is fine. 
- `davinci` CoT performance is crazy.

### 20230116 `GSM8K`
- `main` subset lays out the rationale objectively. 
  - `socratic` subset carries on a series of self qa subjectively. 
  -  Prompt in the paper is more akin to `main` style. 
- When `num_beams=4`, answer = **Janet eats 3 + 4 = 7 eggs a day. She sells 16 - 7 = 11 eggs a day at the farmers' market. She makes 11 * 2 = 22 dollars a day at the farmers' market. The answer is 22.**
  - Even though the answer is wrong. The miss step is 16-7=9, not 11. The logic overall is fine. This is the emergent part of the LLM for reasoning task. So many little intermediate step could go wrong, and on wrong would render the final answer wrong.   
  - It's actually not emergent. To get one reasoning task right, observed from final answer perspective, it's a quantum leap wrt to modele size. However, observed from rationale perpsective, it's not quantum leap and magical anymore. You can observe little mistakes here and there, and they lead to wrong answer. 
  - The other mistake is Janet won't eat 7 eggs, but rather use 7 per day. Semantically, this 3b model got it wrong, even though the number, 7, is the right number for next stage calculation. 
  - Also you can see the beauty of such thinking out loud rationale engineering. It makes abstract computation transparent and **readable** so you could understand and debug incrementally, rather than betting on pure wishful thinking of quantum leap. 
  - When I switch off `num_beams` to `greedy decoding`, it actually generate right answer at 3b scale: **She eats 3 + 4 = 7 eggs every day. She has 16 - 7 = 9 eggs left. She sells 9 * 2 = $18 at the farmers' market. The answer is 18.** Simply amazing. This is better than normal 5 year old reasoning lol. 

## Reference
- [HF dataset process](https://huggingface.co/docs/datasets/process)
