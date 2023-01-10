## Why?
Discord chatbots can serve as effective platforms for testing multimodal agents, with the ultimate goal of creating a helpful, honest, and harmless agent that can be distributed to a wide audience in a self-sustaining manner.  
By using Discord chatbots as a testbed, we can explore ways to build and disseminate useful and trustworthy artificial intelligence.

## `seed` library
The goal is building a modularized AI as starting point. Fairly advanced AI may not need human designed modularization. This setup is better for fast iteration and learning. No worry about human intervention getting in way of AGI. We won't get there in one-shot. Maybe after 1m shot we would be in a position to think about taking off human limitation.

## Discord bot
### Setup
- Build your `.env` from the template `.env.example`. Fill in your tokens. `vscode` would load those env var automatically.
  - [ref: container env](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 
- You need 2x 3090 to run `huggingface` models such as `flan-t5`. Stick to OpenAI api with limited compute budget. 

### Run
```shell
python -m discord_bot.main
```

## Paper reproduction.
### Model selection
- OpenAI `davinci` is the only publicly accessible decorder model that's worth studying. Wait for `anthropic`'s LM. 
- `flan-t5-xxl` is the most versatile and best performant encoder decoder model for now. 
- `t0pp` is weaker, older, open source version of `flan-t5`. Still worth playing with and I could run it locally with. 
  - It has [opensource repo](https://github.com/bigscience-workshop/t-zero) to emulate. Easier to get started. 
  - _instruction finetuning without CoT actually degrades reasoning ability, including just nine CoT datasets improves performance on all evaluations_ from `flan-t5` paper. So reasoning related benchmark on `t0pp` is not worth it. 
```python
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='sequential', torch_dtype=torch.bfloat16)
```

### Paper of intereste
- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](/paper/weiChainThoughtPrompting2022/)
