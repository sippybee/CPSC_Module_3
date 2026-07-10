"""
gpt_train.py
------------
A minimal, from-scratch, decoder-only (GPT-style) character-level language model.

This script assembles the pieces from earlier lessons -- token/positional
embeddings and multi-head self-attention -- into a full, reusable
TransformerBlock, stacks several of those blocks into a small GPT, trains it
on a plain text file (here: article.txt), and generates new text from it
using temperature and top-k sampling.

Run:
    python gpt_train.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
torch.manual_seed(1337)

batch_size    = 32     # how many independent sequences we train on in parallel
block_size    = 128    # maximum context length (in characters) the model sees
n_embd        = 128    # embedding dimension
n_head        = 4      # number of attention heads
n_layer       = 4       # number of TransformerBlocks stacked
dropout       = 0.1
max_iters     = 3000
eval_interval = 300
eval_iters    = 50
learning_rate = 3e-4
device        = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Data: load article.txt and build a character-level tokenizer
# -----------------------------------------------------------------------------
with open("article.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"corpus length: {len(text)} chars, vocab size: {vocab_size}")


def get_batch(split):
    """Grab a random batch of (context, target) sequences from train or val."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Average loss over a few batches, used just for clean logging."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Component: Causal Self-Attention (multi-head), from the previous lesson.
# Included here so the TransformerBlock below is fully self-contained.
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.qkv_proj  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask so each position can only attend to itself and the past
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)                       # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape into (B, n_head, T, head_size) for parallel per-head attention
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v                                # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# -----------------------------------------------------------------------------
# Component: Feed-Forward Network (FFN)
# A position-wise 2-layer MLP applied independently to every token. The 4x
# expansion factor follows the original Transformer / GPT recipe.
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Component: TransformerBlock
# Combines attention + FFN with Layer Normalization and Residual Connections,
# using the GPT-2-style "pre-norm" arrangement: LayerNorm is applied BEFORE
# each sub-layer, and the sub-layer's output is added back (residual) to its
# own input rather than replacing it. This is what lets us stack many blocks
# without the signal (or its gradient) vanishing or exploding.
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection around attention
        x = x + self.ffn(self.ln2(x))    # residual connection around the FFN
        return x


# -----------------------------------------------------------------------------
# The full model: embeddings + stacked TransformerBlocks + output head
# -----------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        # token identity embedding: "what character is this?"
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # positional embedding: "where in the sequence am I?" -- a learned
        # vector per position, added to the token embedding so the otherwise
        # order-blind attention mechanism knows about token order.
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)          # final LayerNorm before the head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb                                            # broadcast add
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B,T,vocab)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively extend `idx` (a (B,T) tensor of token ids) by
        `max_new_tokens`, sampling one character at a time.

        temperature: divides the logits before softmax.
            < 1.0 sharpens the distribution (more confident / repetitive),
            > 1.0 flattens it (more random / creative).
        top_k: if set, restrict sampling to only the k highest-probability
            characters at each step, zeroing out the rest. This trims the
            long tail of unlikely characters that temperature alone can
            still occasionally pick.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # last time step only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# Text generation
# -----------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start from a single "blank" token

print("\n--- greedy-ish sampling, temperature=1.0, no top_k ---")
print(decode(model.generate(context, max_new_tokens=300, temperature=1.0)[0].tolist()))

print("\n--- lower temperature=0.6, top_k=20 (more focused) ---")
print(decode(model.generate(context, max_new_tokens=300, temperature=0.6, top_k=20)[0].tolist()))

print("\n--- higher temperature=1.3, top_k=10 (more varied but still sane) ---")
print(decode(model.generate(context, max_new_tokens=300, temperature=1.3, top_k=10)[0].tolist()))
