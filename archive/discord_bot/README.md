## Setup
- Build your `.env` from the template `.env.example`. Fill in your tokens. `vscode` would load those env var automatically.
  - [ref: container env](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 
- You need serious computation power to run `huggingface` model locally, ex: `t0pp`. Stick to OpenAI api with limited compute budget. 

## Run
```shell
python -m discord_bot.main