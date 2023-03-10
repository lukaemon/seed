## tl;dr
Better zero shot transfer, more sample efficient, minimal catastrophic forgetting. Big model dominates. 
## Context
This is Gato's follow up. Automatic full read status. My hunch is multimodal models learn faster than monomodal models. If that's the case, native multimodal model would render GPT kind of text model as specialized, myopic model. The paradigm shift is immanent. 

That doesn't mean GPT3 is useless. No. Once a model saturates a modality, it's good. However, with ever improving inference optimization and new sparse architecture, LLM may not be relevant anymore if superior multimodal model could perform better at close inference cost. 

## Log
- `Nerf` representation as input is interesting, which is the future real world interaction. Basically an internal 3d world model.
- Task driven feedback loop: "sequential robotic manipulation planning, visual question answering, and captioning." This means no worry about mid attention fusion. `ViT` is a proven image encoder. Just put multimodal tokens together and let the attention to do whatever fusion is necessary wrt tasks. 
- [question -> I don't understand the meaning of text pretraining here. How text only pretrain help multimodal finetuning here? What kind of learned pattern that is transferred?]
- "Inputs such as images and state estimates are embedded into the same latent embedding as language tokens and processed by the self-attention layers of a Transformer-based LLM in the same way as text."
    - So this is grounding! Wow! Force other modality onto text's latent space? I need a day. 
    - In this case, totally make sense to start from PaLM. Is text's latent space a good foundation for other modalities? How to quantify such fusion as good or bad?
- "PaLM-E is trained to generate plans directly without relying on auxiliary models for grounding. This in turn enables direct integration of the rich semantic knowledge stored in pretrained LLMs into the planning process."
    - This is the reason why use the LLM as bedrock. Basically, the world knowledge in LLM is the best option right now before we figure out other ways to learn world knowledge and common sense like baby does.
- `vocab_size` = 256k, `n_embd` = 18432
- MLP map everything to LLM's latent space. All you need is one MLP. ViT only needs an even simpler affine transformation. Recurring theme @merulloLinearlyMappingImage2022. 
- [eureka -> 8b LLM + 4b Vit. I probably could reproduce this with 6.7b LLaMA + 4b ViT. Or even try smaller footprint such as T5-xl + smaller ViT.]
- This paper is locked LLM with unlocked ViT. LiT, @zhaiLiTZeroShotTransfer2022 is locked ViT with unlocked LLM. Multimodal representation learning is unsolved. I think visual-audio pretraining with sporadic language supervision to bootstrap, then all in supervised language learning is the way to go. Basically reproduces the human curriculum.
- "The notable trend is that with increasing model scale, there is considerably less catastrophic forgetting of language capabilities."
    - ![](f6.png)
    - This is critical and hard to interpret. Because the finetuning maps multimodal representation to originally text pretrained latent space, catastrophic forgetting is expected. Larger model could retain the most of original performance is amazing. With advanced finetuning techniques, @ansellComposableSparseFineTuning2022, maybe one could revive dead zones to host new skills without catastrophic forgetting.
    - [lesson_learned -> embodied intelligence with sufficient model size would dominate text only model. Text alone is not defensible. Have to keep climbing the modality ladder.]
- "PaLM-E trained on different tasks and datasets at the same time leads to significantly increased performance relative to models trained separately on the different tasks alone."
    - Doom of specialized small model. Larger, multimodal model simply learns faster and transfer better. One could always distill, but in terms of raw intelligence, multimodal large model is the way to go.
    - [lesson_learned -> pretraining is still necessary for solo modality initialization. But multimodal multitask finetuning is how to scale IQ up effectively.]
- Maybe `OSRT` is better representation than images. Human experiences are mostly grounded in 3d world. Image is like text, a kind of abstraction, human artifacts. 