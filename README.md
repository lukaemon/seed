## Why?
Discord chatbot is an effective platform for testing multimodal agents, with the ultimate goal of creating a helpful, honest, and harmless AI that can be distributed to wide audience in a self-sustaining manner.  
By using Discord chatbot as testbed, we can explore ways to build and disseminate useful and trustworthy AI.

## `seed` library
The goal is building a modularized AI as starting point. Fairly advanced AI may not need human designed modularization. The setup is for fast and cheap failures to facilitate effective learning. 

Learnings from paper reproduction and project would trickle back to the `seed` library.

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

## Create
![](asset/copy_transform_combined.jpeg)
Everything is a Remix: [part1](https://www.youtube.com/watch?v=MZ2GuvUWaP8) | [part2](https://www.youtube.com/watch?v=HhMar_eYnNY) | [part3](https://www.youtube.com/watch?v=dwxtW1Aio68). This is open souce ethos, and the future is counting on open source AI.  

### Model Zoo
- OpenAI `davinci` is great decorder model to study for cutting edge LLM capability.
- `UL2` could be a weak local alternative to `davinci`.
- `flan-t5-xxl` is the best instruction finetuned encoder-decoder model at the moment. 
- `t0pp` is an instruction finetuned model from huggingface. [repo](https://github.com/bigscience-workshop/t-zero), a good starting point. 

### Works
- Basic
  - [Multitask Prompted Training Enables Zero-Shot Task Generalization](paper/sanhMultitaskPromptedTraining2022a/)
- Pretrain model
- Finetuninig
  - [Scaling Instruction-Finetuned Language Models](paper/chungScalingInstructionFinetunedLanguage2022)
- Rationale engineering (think)
  - [Chain of Thought Prompting Elicits Reasoning in Large Language Models](paper/weiChainThoughtPrompting2022/)
  - [Self-Consistency Improves Chain of Thought Reasoning in Language Models](paper/wangSelfConsistencyImprovesChain2022a)
  - [Recitation-Augmented Language Models](paper/sunRecitationAugmentedLanguageModels2022a)
- Do something (act)
