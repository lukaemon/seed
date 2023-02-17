> Fail diversely, fail fast, and iterate.

## Why?
I see cutting edge AI paper getting published and real world production deployment at light speed in the past 3 years. Stunning. Brilliant idea, great execution, huge improvement. Exciting and hopeful of all. 

However, for people to truly adopt AI and cultivate capability to use, cooperate and even create one with autonomy, the embarrassing side of the learning process may be of some value. I personally would love to see more fail experiment, detour and dead end, as well as the rosy final result.

## How to use this repo?
Maybe the best way to interact with this repo is by conditioning the ChatGPT with it and just talk. ex: 
- List all project name. Categorize by topics and counting sort for me. 
- Find all NLP projects that only deal with prompts. I don't want to do any training for now. 
- What project did he do first? What did he learn?
- Gauge his level of expertise given the first project. 
- What mistakes are made then?
- Did he learn from it and improve on the second project? 
- How can he do better on the second project? Could it be better coding, choice of problem, experimental design or general critical thinking?
- Design a curriculum for me. The goal is to reach his level of expertise around project 2 asap. Tailored to my applied field xyz and current level of expertise. Take my github and personal knowledge base as the baseline evaluation of me. 
- Make the curriculum more hands on, less theoretic and more interactive. 

Think of this repo as an unstructured database and ChatGPT as the rudimentary UI/UX.

For direct readers, progressive details per project from tl;dr to log are available. Check [project](#project).

## Code Environment
- Get familiar with [VScode dev container](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file). 
- Build `.env` from the template `.env.example`. Fill in tokens. `vscode` would load env var automatically.
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`.

ps: serious computation power required to run model locally, ex: `flan-t5-xxl`. Choose smaller model or stick to OpenAI api with limited compute budget.  

## Project
Each project is atomic, self-contained brand new start. At this stage I hope to only carry over ideas, not codes or prescribed structure. Repetition on purpose. Sample new way of doing new things with least legacy. When mature enough, isolated library would emerge to reduce unnecessary overhead. 

Project `README.md` will present:
### tl;dr
Elevator pitch of the project.

### Context
Why? What triggers? To what ends?

### Done
What is actually created? 

### Learned
Answered known unknown. Realized unknown unknown. Connect the dots. 

### Next?
Build up leads during the project. Don't care about the quality of idea. Record every light bulb moment. Critique at the end.
- Possible incremental, logical improvements. 
- Possible quantum leap, intuitive, creative, even crazy ideas to try. 

Important questions from Log would be placed here. 

### Log 
Operational log. Event driven, append only and chronologically sorted. Lightly grouped by topic since sub-lists are direct, immediate expansion of parent point. Consider top level list as trigger prompts. Chronological flow of thoughts.
- Positive: making break through, good result, intuition, execution. 
- Negative: dead end, unproductive loop, failed experiment, tentative, flawed chain of thought, even emotional battle, ex: frustration, despair, giving up. 

Overall it's similar to captain's log or super verbose chain of thought. Not the most human friendly format but I assume 99% of human won't actually read logs. The target audience is AI.

## Trajectory
Check the [google sheet](https://docs.google.com/spreadsheets/d/11Ul6yh4x3HCz35SVBTQCOFwBEhI2CHr9H9a-CAggP6g/edit?usp=sharing) for the trajectory. Read [learning to learn](https://lukaemon.github.io/posts/2023/learning-to-learn/) for why I structure the trajectory and project this way.

