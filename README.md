> Road to helful, harmless and helpful agent.

# Feature

# Setup
- Build your `.devcontainer/devcontainer.env` from the template `.env.example`. 
  - [ref: container evn](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 