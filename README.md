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
> Create to ask new question, and old question sharper. 

![](asset/copy_transform_combined.jpeg)

### Model selection
- OpenAI `davinci` is the only publicly accessible decorder model that's worth studying for cutting edge capability of LLM. Wait for `anthropic`'s API, if it's going to be released at all. 
- `UL2` could be local alternative to `davinci` for the poor. Don't expect on par CoT reasoning capability.
- `flan-t5-xxl` is the best instruction finetuned open encoder-decoder model at the moment. 
- `t0pp` is weaker `flan-t5`. It's [github](https://github.com/bigscience-workshop/t-zero) code is good starting point. 

### Category
- Basic
  - [Multitask Prompted Training Enables Zero-Shot Task Generalization](paper/sanhMultitaskPromptedTraining2022a/) - (2022)
- Base model
- Finetuninig
  - [Scaling Instruction-Finetuned Language Models](paper/chungScalingInstructionFinetunedLanguage2022) - (2022)
- Rationale engineering (think)
  - [Chain of Thought Prompting Elicits Reasoning in Large Language Models](paper/weiChainThoughtPrompting2022/) - (2022)
  - [Self-Consistency Improves Chain of Thought Reasoning in Language Models](paper/wangSelfConsistencyImprovesChain2022a) - (2022)
  - [Recitation-Augmented Language Models](paper/sunRecitationAugmentedLanguageModels2022a) - (2022)
- Do something (act)
