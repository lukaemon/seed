## What I've done?
- `gsm8k` on `flan-t5-xxl` and `davinci`. Check notebook for results.
- It's too expensive for full evaluation reproduction. I'll stop here. 


## What I've learned?
- Self-consistency is not effective on t5 11b model. Understandable. `Reasoning` is an emergent capability that usually show after `60b` scale. The paper is on `PaLM 540b`. 
- Use **voting as measurement of uncertainty** is one great realization. This is critical first step for LLM calibration. 
- Many answer parsing error on finetuned `davinci` model response. The answer is correct but the simple regex won't get it right. The answer is too good for human LoL. 
- Both `SC` and `CoT` are methods of scaling computation to solve problems. Similar to think harder, longer, with scratchpad for important or hard problems. 
- The cost is substantially increased, especially for `SC`. 
- Cutting edge model is only feasible to apply on high value, mission critical problems. Cost/Reward analysis is important to make a viable business plan.
- For simple, low value tasks, you have to train/ft small models, or use lower tier api, ex:`curie`. 
- I don't know which way is the right way to go.
  - One could go [Neeva](https://neeva.com/) route, consumer facing, tight vertical intergration, end 2 end model, UI/UX, payment. The strategy is simple: 
    - Go after big and validated consumer market.
    - Give up LLM API. Cook in house LLM, aiming for 10x experience, and own the customer experience end2end to secure the data flywheel. 
  - One could go [Isomorphic Labs](https://www.isomorphiclabs.com/) route, cutting edge, super niche, super high value. 
  - However, I don't think [Elicit](https://elicit.org/) model could work. Weird middle ground such that the value is not high enough to warrant cutting edge api cost, but not general enough to capture fraction of possible TAM. I just don't know. 
  - Unfortunately, both Neeva, and Iso are extremely capital intensive. I'm still not sure if AI is startup friendly. 

## Log
- OpenAI rate limit reached on `code-davinci-002`. 
  - That's why you need to curate and own the intelligence. 
  - It could be rate limited, banned, sensored without notice. 
```python
RateLimitError: Rate limit reached for default-code-davinci-002 in organization org-dWeZbE95G8RwWn7NnBbrKqnW on tokens per min. Limit: 40000.000000 / min. Current: 40960.000000 / min. Contact support@openai.com if you continue to have issues.
```