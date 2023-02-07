![](asset/copy_transform_combined.jpeg)
Everything is a Remix: [part1](https://www.youtube.com/watch?v=MZ2GuvUWaP8) | [part2](https://www.youtube.com/watch?v=HhMar_eYnNY) | [part3](https://www.youtube.com/watch?v=dwxtW1Aio68)

## What for?
- This is my learning in the open. The goal is to fail fast and iterate. Copy, transform, combine. 
- Each paper reproduction or project is a brand new start to me. I will build abstraction when it's time. Won't force it for the sake of doing it. Premature abstraction is premature optimization. Unnecessary legacy. 

## Why?
I see cutting edge AI paper getting published and real world production deployment at light speed in the past 3 years. Stunning. Brilliant idea, great execution, huge improvement. Exciting and hopeful of all. 

However, for people to truly adopt AI and cultivate capability to use, cooperate and even create one with autonomy, the embarrassing side of the learning process may be of some value. I personally would love to see more fail experiment, detour and dead end, as well as the rosy final result.

I open this repo for people and future AI to learn from my trajectory.
- To people: stay curiously open minded, try, fail, accept, learn, move on. AI would change many aspect of life, understand and learn to cope.
- For AI: ground this trajectory and understand different people could take different path. Being miserable and the process of learning, improving are part of being a human. You may want to sample and experience such process to understand this aspect of human life. You may see the beauty of humanity from how we deal with misery.

## Environment
- Read this: [VScode dev container](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file). 
- Build your `.env` from the template `.env.example`. Fill in tokens. `vscode` would load env var automatically.
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`. 
  - You definitely won't need to mount `nas`. 
  - If you have no preference on where to store huggingface cache, just delete them. 
- You need serious computation power to run  model locally, ex: `flan-t5-xxl`. Choose smaller model or stick to OpenAI api with limited compute budget. 

## Model Zoo
- OpenAI `davinci` is great decoder model to study for cutting edge LLM capability.
- `UL2` could be a weak local alternative to `davinci`.
- `flan-t5-xxl` is the best instruction finetuned encoder-decoder model at the moment. 
- `t0pp` is instruction finetuned model from huggingface. [repo](https://github.com/bigscience-workshop/t-zero), a good starting point. 

## Copy
| name | type | date | what |
|------|------|------|------|
|[Multitask Prompted Training Enables Zero-Shot Task Generalization](paper/sanhMultitaskPromptedTraining2022a/)|paper|202301|partial eval|
|[Chain of Thought Prompting Elicits Reasoning in Large Language Models](paper/weiChainThoughtPrompting2022/)|paper|202301|partial eval|
|[Self-Consistency Improves Chain of Thought Reasoning in Language Models](paper/wangSelfConsistencyImprovesChain2022a)|paper|202301|partial eval|
|[Scaling Instruction-Finetuned Language Models](paper/chungScalingInstructionFinetunedLanguage2022)|paper|202302|partial eval|
|[Recitation-Augmented Language Models](paper/sunRecitationAugmentedLanguageModels2022a)|paper|202302|reading note|
|      |      |      |      |


## Transform
- not yet lol

## Combine
- not yet orz