# %%
import os
import numpy as np
import torch

import config

# %%
train_path = os.path.join(os.path.dirname(__file__), "train.bin")
train_data = np.memmap(train_path, np.uint16, "r")

val_path = os.path.join(os.path.dirname(__file__), "val.bin")
val_data = np.memmap(val_path, np.uint16, "r")

# %%
def get_batch(split):
    """
    The data is stored as unsigned int 16 since the vocab_size < 65536
    Transform the numpy to np.int64 since the model is expecting torch.long (int64)
    """
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - config.block_size, size=(config.batch_size,))

    x = torch.stack(
        [
            torch.from_numpy(data[i : i + config.block_size].astype(np.int64))
            for i in idx
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + config.block_size].astype(np.int64))
            for i in idx
        ]
    )

    x = x.pin_memory().to(config.device, non_blocking=True)
    y = y.pin_memory().to(config.device, non_blocking=True)

    return x, y


# %%
