## Why?
Discord chatbot is an effective platform for testing multimodal agents, with the ultimate goal of creating a helpful, honest, and harmless AI that can be distributed to wide audience in a self-sustaining manner.  
By using Discord chatbot as testbed, we can explore ways to build and disseminate useful and trustworthy AI.

## `seed` library
The goal is building a modularized AI as starting point. Fairly advanced AI may not need human designed modularization. The setup is for fast and cheap failures to facilitate fast learning. 

## Discord bot
### Setup
- Build your `.env` from the template `.env.example`. Fill in your tokens. `vscode` would load those env var automatically.
  - [ref: container env](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 
- You need serious computation power to run `huggingface` model locally, ex: `t0pp`. Stick to OpenAI api with limited compute budget. 

### Run
```shell
python -m discord_bot.main
```

## Paper reproduction
### Model selection
- OpenAI `davinci` is the only publicly accessible decorder model that's worth studying. Wait for `anthropic`'s LM. 
- `flan-t5-xxl` is the most versatile and best performant encoder decoder model for now. 
- `t0pp` is weaker, older, open source version of `flan-t5`. Still worth playing with. 
  - [open source repo](https://github.com/bigscience-workshop/t-zero). Great starting point. 
  - _instruction finetuning without CoT actually degrades reasoning ability, including just nine CoT datasets improves performance on all evaluations_ from `flan-t5` paper. So reasoning related benchmark on `t0pp` is detrimental. 

### List
- [Multitask Prompted Training Enables Zero-Shot Task Generalization (202203)](/paper/sanhMultitaskPromptedTraining2022a/)
- [Chain of Thought Prompting Elicits Reasoning in Large Language Models (202210)](/paper/weiChainThoughtPrompting2022/)
