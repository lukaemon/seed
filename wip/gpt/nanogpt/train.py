"""difference to chargpt
✅ loss fn
    - reduction='mean' by default. 
✅ compile model
✅ mix precision training
✅ gradient clip
✅ gradient accumulation
✅ ddp
- estimate model flop utilization
- wandb logging
- lr scheduler
- weight decay
- checkpointing and resume
"""


import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import fire
from tqdm import tqdm

import load_data
from model import GPT
from config import GPT2Config


def loss_fn(logits, target):
    """
    logits: (b, t, vocab_size)
    target: (b, t)
    """
    b, t, c = logits.shape
    return F.cross_entropy(logits.view(-1, c), target.view(-1))


cfg = GPT2Config()
is_ddp = os.getenv("RANK") is not None

if is_ddp:
    init_process_group(backend=cfg.backend)
    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    device = f"cuda:{local_rank}"
    is_master = rank == 0
    seed_offset = rank
else:
    is_master = True
    seed_offset = 0
    device = "cuda"


torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

model = GPT(cfg).to(device)
model = torch.compile(model)

if is_ddp:
    model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def eval():
    model.eval()
    loss = {}

    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)

        for iter in range(cfg.eval_iters):
            x, y = load_data.get_batch(split, device)
            losses[iter] = loss_fn(model(x), y).item()

        loss[split] = losses.mean()
    model.train()

    return loss


start = time.time()
x, y = load_data.get_batch("train", device)


for iter in tqdm(range(cfg.max_iters)):
    if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
        loss = eval()
        print(f"{iter=} | train_loss={loss['train']:.3f} | val_loss={loss['val']:.3f}")

    # gradient accumulation
    # just keep doing forward backward without optimizer step gradient would accumulate
    for micro_step in range(cfg.gradient_accumulation_steps):
        if is_ddp:  # gradient sync at the last micro step when ddp
            model.require_backward_grad_sync = (
                micro_step == cfg.gradient_accumulation_steps - 1
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)

        x, y = load_data.get_batch("train", device)  # async prefetch next batch
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

time_span = time.time() - start
effective_n_batch = cfg.max_iters * cfg.gradient_accumulation_steps
total_tokens = effective_n_batch * cfg.batch_size * cfg.block_size
print(f"Training took {time_span:.2f} seconds.")
print(f"Throughput per node: {total_tokens / time_span:.2f} token/s")

if is_ddp:
    destroy_process_group()
