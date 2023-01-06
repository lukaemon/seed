## Why?
Discord chatbots can serve as effective platforms for testing multimodal agents, with the ultimate goal of creating a helpful, honest, and harmless agent that can be distributed to a wide audience in a self-sustaining manner.  
By using Discord chatbots as a testbed, we can explore ways to build and disseminate useful and trustworthy artificial intelligence.

## `seed` library
The goal is building a modularized AI as starting point. Fairly advanced AI may not need human designed modularization. This setup better for fast iteration and learning. No worry about human intervention getting in way of AGI. We won't get there in one-shot. Maybe after 1m shot we would be in a position to think about taking off human limitation.

## Discord bot
### Setup
- Build your `.devcontainer/devcontainer.env` from the template `.env.example`. 
  - [ref: container env](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 
- You need 2x and more GPUs to run `huggingface` models. Stick to OpenAI api with limited compute budget. 

### Run
```shell
python -m discord_bot.main
```