"""still using stocking data
- scrape
- clean
- tokenize
- dataloader for training

Basically you want to process the whole openwebtext into a long list of int. 
"""
# %%
import os
import tiktoken
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# %%
ds = load_dataset("openwebtext", split="train").train_test_split(
    test_size=0.0005, shuffle=True, seed=42
)
ds["val"] = ds.pop("test")

# %%
tk = tiktoken.get_encoding("gpt2")


# %%
def process(batch):
    ids = tk.encode_ordinary_batch(batch["text"])
    n_token = [len(x) for x in ids]

    return {"ids": ids, "n_token": n_token}


# %%
ds = ds.map(process, remove_columns=["text"], num_proc=16, batched=True)


# %%
for split, dset in ds.items():
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    shape = np.sum(dset["n_token"])

    print(f"{split} set has {shape} tokens.")

    mmap = np.memmap(filename, np.uint16, mode="w+", shape=shape)

    idx = 0
    for example in tqdm(dset):
        mmap[idx : idx + example["n_token"]] = example["ids"]
        idx += example["n_token"]

    mmap.flush()


# train ~9b tokens
# val ~4m tokens

# %%
# val_path = os.path.join(os.path.dirname(__file__), "val.bin")
# val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
# len(val_data)
# val_data[:100]
