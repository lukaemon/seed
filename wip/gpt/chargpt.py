# %%
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# hyperparameter
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
p_drop = 0.2
# ------------

torch.manual_seed(1337)

with open("/workspaces/seed/wip/gpt/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# %%
class Head(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        b, t, c = x.shape

        q = self.query(x)  # (b, t, hs)
        k = self.key(x)
        v = self.value(x)

        wei = F.softmax(q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5, dim=-1)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (b, t, t)
        wei = self.dropout(wei)

        out = wei @ v  # (b, t, hs)

        return out


# %%
class MultiheadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_head) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out


# %%
class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


# %%
class Layer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.mha = MultiheadAttention(n_embd, n_head)
        self.ffn = FFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# %%
class GPT(nn.Module):
    def __init__(self, n_embd, n_head, n_layer) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.stack = nn.Sequential(*[Layer(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, idx, target=None):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)  # (b, t, c)
        pos_emb = self.position_embedding(torch.arange(t, device=device))  # (t, c)

        x = self.dropout(tok_emb + pos_emb)
        x = self.stack(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        if target is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            target = target.view(b * t)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_length):
        for _ in range(max_length):
            context = idx[:, -block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]  # (b, c)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# %%
model = GPT(n_embd, n_head, n_layer)
model = model.to(device)
model = torch.compile(model)

print(f"Number of parameters = {sum(p.numel() for p in model.parameters()) / 1e6}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training took {time.time() - start:.2f} seconds")
# %%
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_length=500)[0].tolist()))

# val loss = 1.83, 764 sec, vanilla pytorch

# default torch.compile, 632.6 sec
# with:
# torch.set_float32_matmul_precision("high")
# torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# training time down to 487 sec, loss = 1.66, 56% faster

# mode="max-autotune", dynamic=False, fullgraph=True, 473 sec. Default setting is fine.
