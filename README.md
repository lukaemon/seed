# Why?
Discord chatbots can serve as effective platforms for testing multimodal agents, with the ultimate goal of creating a helpful, honest, and harmless agent that can be distributed to a wide audience in a self-sustaining manner. By using Discord chatbots as a testbed, we can explore ways to build and disseminate useful and trustworthy artificial intelligence that can benefit people around the world.

# Setup
- Build your `.devcontainer/devcontainer.env` from the template `.env.example`. 
  - [ref: container evn](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file)
  - [ref: discord bot token?](https://github.com/openai/gpt-discord-bot#setup)
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on huggingface cache, just delete them. 