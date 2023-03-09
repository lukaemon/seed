# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# ------------

torch.manual_seed(1337)

# %%
with open("/workspaces/seed/wip/gpt/input.txt") as f:
    text = f.read()

len(text)

# %%
chars = sorted(set(text))
# print("".join(chars))

# %%
vocab_size = len(chars)
vocab_size

# %%
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}

encode = lambda t: [ctoi[c] for c in t]
decode = lambda idx: "".join([itoc[i] for i in idx])

# %%
decode(encode("hello, world!"))

# %%
data = torch.tensor(encode(text))
data.shape

# %%
n = int(len(data) * 0.9)  # split
train_data = data[:n]
val_data = data[n:]

assert len(train_data) + len(val_data) == len(data)

# %%
def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack(
        [data[i + 1 : i + 1 + block_size] for i in ix]
    )  # shift to the right

    x, y = x.cuda(), y.cuda()
    return x, y


# %%
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.table = nn.Embedding(vocab_size, vocab_size)  # (vocab_size, c)

    def forward(self, idx, target=None):
        logits = self.table(idx)  # (b, t, c)

        if target is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            target = target.view(b * t)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_length):
        """batch generate
        idx: (b, t) as batch context
        """
        for _ in range(max_length):
            idx_cont = idx[:, -block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (b, c)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (b, t+1)

        return idx


# %%
model = BigramLM(vocab_size)
m = model.cuda()

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# %%
@torch.no_grad()
def eval_loss():
    m.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = m(x, y)
            losses[i] = loss.item()

        out[split] = losses.mean()
    m.train()
    return out


# %%
for iter in range(max_iters):
    if iter % eval_interval == 0:
        out = eval_loss()
        print(f"step {iter}, train loss {out['train']:.4f}, val loss {out['val']:.4f} ")

    x, y = get_batch("train")
    logits, loss = m(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
context = torch.zeros((1, 1), dtype=torch.long).cuda()
print(decode(m.generate(context, 500)[0].tolist()))
# %%

# val loss = 2.49
