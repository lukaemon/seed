from dataclasses import dataclass


@dataclass
class GPT2Config:
    out_dir = "out"
    max_iters = 100  # 600000  # total number of training iterations
    eval_interval = 100  # 500
    eval_iters = 10  # 200
    log_interval = 1
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = "gpt2"

    # data
    gradient_accumulation_steps = 8  # used to simulate larger batch sizes
    batch_size = 16
    block_size = 1024
    vocab_size = 50304

    # model
    n_layer = 12  # 12
    n_head = 12  # 12
    n_embd = 768  # 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?

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

    # system
    device = "cuda"
