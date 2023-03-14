from dataclasses import dataclass


@dataclass
class GPT2Config:
    # data
    max_iters = 100  # 600000  # total number of training iterations
    batch_size = 16
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    block_size = 1024

    eval_interval = 100  # 500
    eval_iters = 10  # 200

    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    vocab_size = 50304

    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.

    # checkpoint
    out_dir = "out"
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    log_interval = 1
    wandb_log = False  # disabled by default
    wandb_project = "gpt2"
