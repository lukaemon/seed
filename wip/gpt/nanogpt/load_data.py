# %%
import os
import numpy as np
import torch

from config import GPT2Config

cfg = GPT2Config()

# %%
train_path = os.path.join(os.path.dirname(__file__), "train.bin")
train_data = np.memmap(train_path, np.uint16, "r")

val_path = os.path.join(os.path.dirname(__file__), "val.bin")
val_data = np.memmap(val_path, np.uint16, "r")

# %%
def get_batch(split, device):
    """
    The data is stored as unsigned int 16 since the vocab_size < 65536
    Transform the numpy to np.int64 since the model is expecting torch.long (int64)
    """
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - cfg.block_size, size=(cfg.batch_size,))

    x = torch.stack(
        [torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in idx]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64))
            for i in idx
        ]
    )

    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)

    return x, y


# %%
