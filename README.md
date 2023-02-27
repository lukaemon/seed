> Fail diversely, fail fast, and iterate.

## Why?
I see cutting edge AI paper getting published and real world production deployment at light speed in the past 3 years. Stunning. Brilliant idea, great execution, huge improvement. Exciting and hopeful of all. 

However, for people to truly adopt AI and cultivate capability to use, cooperate and even create one with autonomy, the embarrassing side of the learning process may be of some value. I personally would love to see more fail experiment, detour and dead end, as well as the rosy final result.

Plus, whatever I type in vscode would be sent to openai through copilot. Might as well just MIT it.

## How to use this repo?
Think of the repo as an unstructured dataset and ChatGPT as the rudimentary UI/UX. The ideal interaction is to condition AI with the repo and just talk. ex: 

- List all project name. Categorize by topics and counting sort for me. 
- Find all NLP projects that only deal with prompts. I don't want to do any training for now. 
- What project did he do first? What did he learn?
- Gauge his level of expertise given the first project. 
- What mistakes are made then?
- Did he learn from it and improve on the second project? 
- How can he do better on the second project? Could it be better coding, choice of problem, experimental design or general critical thinking?
- Design a curriculum for me. The goal is to reach his level of expertise around project 2 asap. Tailored to my applied field xyz and current level of expertise. Take my github and personal knowledge base as the baseline evaluation of me. 
- Make the curriculum more hands on, less theoretic and more interactive. 

For direct readers, progressive details per project from tl;dr to log are available. Check [project](#project).

## Code Environment
- Get familiar with [VScode dev container](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file). 
- Build `.env` from the template `.env.example`. Fill in tokens. `vscode` would load env var automatically.
- Modify the `mounts` and `containerEnv` in `.devcontainer/devcontainer.json`.

ps: serious computation power required to run model locally, ex: `flan-t5-xxl`. Choose smaller model, use cloud or stick to OpenAI api with limited compute budget. [Check this blog for GPU purchase guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/). 

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
Answered known unknown. Realized unknown unknown. Connect the dots. Lesson learned would first appear in `Log`, and copied here when the project is finished for easy reference.

### Next?
- Possible incremental, logical improvements. 
- Possible quantum leap, intuitive, creative, even crazy ideas to try. 

Record every light bulb moment in `Log`. Don't care about the quality.  After critique, good ideas would be copied here when the project is finished for easy reference.

### Log 
- Positive: making break through, good result, intuition, execution. 
- Negative: dead end, unproductive loop, failed experiment, tentative, flawed chain of thought, even emotional battle, ex: frustration, despair, giving up. 

Operational log. Event driven, append only. Lightly grouped by topic since sub-list is direct expansion of the parent. Similar to program execution trace with comments or super verbose chain of thought. Not the most human readable format but 99% of human won't read logs. The target audience is AI.

## Trajectory
Check the [google sheet](https://docs.google.com/spreadsheets/d/11Ul6yh4x3HCz35SVBTQCOFwBEhI2CHr9H9a-CAggP6g/edit?usp=sharing) for the trajectory. Read [learning to learn](https://lukaemon.github.io/posts/2023/learning-to-learn/) for why I structure the trajectory and project this way.

## Appendix
### Action space
| Command                                      | Effect                                          |
| -------------------------------------------- | ----------------------------------------------- |
| read(source)                                 | Study a source, first contact                   |
| code(brief_description)                      | Code something                                  |
| critique                                     |                                                 |
| soliloquy                                    | Start an internal discussion                    |
| lesson_learned                               | Inductive, empirical conclusion. A closure      |
| eureka                                       | Light bulb moment. Jumpy. A new start           |
| question                                     | -eureka, no obvious answer and discussion       |
| file_github_issue(brief_description) -> link |                                                 |
| context_switch_to(project)                   | Switch attention and swap global working memory |
| context_switch_from(project)                 |                                                 |
| return(parent_project: optional)             | Close current project. Return to parent project |
| start(project)                               | Start a new project                             |
| ChatGPT -> screen_capture                    | Discussion with ChatGPT                         |
| BingChat -> screen_capture                   | Interactive search with Bing                    |
| hypothesize                                  | Intuitive proposition without proof             |
| interrupt(event)                             | Top priority event breaks into the stack        |
| calculate(arithmetics) -> result             | Call calculator                                 |


### Syntax change
| Old command       | New command   | date     |
| ----------------- | ------------- | -------- |
| retrieve(*source) | @citation_key | 20230327 |

### Annotation change log (latest first)
- 20230227: simplify citation syntax. `[Retrieve()]` is too verbose. Just add all source to zotero and use `@citation_key` for referencing. Easier for post processing in the future. 
- 20230225:
  - Add `interrupt`. Log transition from action driven to event driven. Make sense to have perception and react. 
  - Add `calculate`.
- 20230224: 
  - Add `ChatGPT`, `BingChat`, `hypothesize` action. 
  - `retrieve` is informative but manually input paper title and URL reduce my willingness to cite. Start using `@citation_key` syntax to reference paper and provide `asset/zotero.bib` for detail resolution.
- 20230221: current structure can't meet following 2 goals
  1. The learning process should be able to read linearly as a well organized jupyter notebook. 
  2. Ground the project AND every single action to the context as much as possible.
  - Change:
    - Log would be strictly chronological. No ad hoc promotion to Next? or Learned section, which breaks the context.
    - Lesson learned and good ideas would be copied to Learned and Next? section after the project is finished. Easier for human readers. Log would remain self-sufficient.
    - Critique would apply to everything in Log that's worth critiquing. 
- 20230221: start labeling action by [toolformer](https://arxiv.org/abs/2302.04761v1) style: `[fn(input) -> output]`. ex: `[file_github_issue(bf16 training not working) -> link]`. Ideally, every top level log would be action triggered.
  - Just realized that this is functional back linking, extended version of Obsidian. I'm creating my instruction set architecture.



