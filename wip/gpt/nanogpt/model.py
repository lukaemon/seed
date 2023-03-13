"""difference to chargpt
✅ bias free layernorm
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html?highlight=layer_norm#torch.nn.functional.layer_norm

✅ bias free linear

✅ batched multihead attention: with built in flash attention implementation

✅ careful with dropout position
  - output of every sub layer
  - output of token embedding + positional embedding

✅ vocab size pad to closest 64: 50304

✅ weight tying

- weight init

- counting number of params

- generate sequence
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPT2Config

# %%
class LayerNorm(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


# %%
class CausalMultiheadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.output_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):  # (b, t, c)
        b, t, c = x.shape

        # (b, t, 3c) -> (b, t, c)
        q, k, v = torch.split(self.input_proj(x), self.config.n_embd, dim=-1)

        # (b, t, nh, hs) -> (b, nh, t, hs)
        q = q.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)
        k = k.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)
        v = v.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(  # (b, nh, t, hs)
            q, k, v, attn_mask=None, dropout_p=self.config.dropout, is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.output_proj(out)
        out = self.dropout(out)

        return out


# %%
class FFN(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.ln2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.ln2(x)

        x = self.dropout(x)

        return x


# %%
class Layer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = LayerNorm(config)
        self.mha = CausalMultiheadAttention(config)
        self.ln2 = LayerNorm(config)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


# %%
class GPT(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.Sequential(*[Layer(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        self.lm_head.weight = self.token_embedding.weight  # weight tying

    def forward(self, ids):
        b, t = ids.shape
        assert t <= self.config.block_size, "input ids length > block_size"

        te = self.token_embedding(ids)  # (b,t,c)
        pe = self.position_embedding(  # (t,c)
            torch.arange(t, dtype=torch.long, device=self.config.device)
        )

        x = self.dropout(te + pe)
        x = self.layers(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
