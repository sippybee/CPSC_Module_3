# The Full Transformer Block & Text Generation

**Goal:** assemble token/positional embeddings and multi-head attention (built in earlier lessons) into a reusable `TransformerBlock`, stack several of them into a small GPT-style decoder, train it on a plain text file, and generate new text with temperature and top-k sampling.

For a training corpus we'll use the uploaded `article.txt` (a magazine feature, ~30,789 characters, 82 unique characters) instead of the usual tiny-Shakespeare file — same mechanics, different flavor of text. All the code below lives in the companion file `gpt_train.py`; this document walks through *why* each piece is built the way it is.

---

## 1. Layer Normalization

Every sub-layer in the block is preceded by an `nn.LayerNorm`. LayerNorm rescales each token's feature vector to zero mean / unit variance (with learned scale and shift), independently per token. In a stack of many blocks this keeps activations in a well-behaved range so gradients don't blow up or shrink as they flow backward through dozens of layers.

We use the **pre-norm** convention (normalize *before* the sub-layer, not after), which is what GPT-2 and most modern decoders use — it trains more stably than the original 2017 Transformer's post-norm design:

```python
self.ln1 = nn.LayerNorm(n_embd)
self.ln2 = nn.LayerNorm(n_embd)
```

## 2. Residual Connections

Each sub-layer's output is *added back* to its own input rather than replacing it:

```python
x = x + self.attn(self.ln1(x))   # residual around attention
x = x + self.ffn(self.ln2(x))    # residual around the FFN
```

This gives gradients a direct path ("highway") straight back to earlier layers, independent of how deep the network is. Without it, stacking more than a handful of blocks becomes very hard to train.

## 3. Feed-Forward Network (FFN)

After attention lets tokens exchange information, the FFN processes each token's vector independently — it's the part of the block that does per-token "thinking":

```python
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
```

The 4x expansion (`n_embd -> 4*n_embd -> n_embd`) is the same ratio used in the original Transformer and in GPT-2/3 — it gives the layer enough capacity to combine features in nonlinear ways before compressing back down.

## 4. Positional Encodings

Self-attention has no built-in sense of order — it treats the input as a set, not a sequence. We fix that with a **learned positional embedding**, a second embedding table (`block_size` rows, one per possible position) that gets added to the token embedding:

```python
self.token_embedding    = nn.Embedding(vocab_size, n_embd)
self.position_embedding = nn.Embedding(block_size, n_embd)

...
tok_emb = self.token_embedding(idx)                                    # (B,T,C)
pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
x = tok_emb + pos_emb
```

This is the GPT-style choice (a learned table) rather than the fixed sinusoidal formula from the original Transformer paper — simpler to implement and works well at this scale.

---

## Architecture: The TransformerBlock

Putting Layer Norm, attention, residuals, and the FFN together gives one reusable block:

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffn  = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

Because the block's input and output shapes match `(B, T, n_embd)` exactly, we can stack as many as we like with a single line:

```python
self.blocks = nn.Sequential(*[
    TransformerBlock(n_embd, n_head, block_size, dropout)
    for _ in range(n_layer)
])
```

The full model is then: token embedding + positional embedding → N stacked blocks → final `LayerNorm` → a linear "language modeling head" that projects back up to vocabulary size for next-character prediction.

---

## Coding Goal: Training Loop

With the model defined, training is an ordinary supervised loop: sample random chunks of `article.txt`, predict the next character at every position, and minimize cross-entropy loss.

```python
data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = data[:n], data[n:]     # 90/10 split

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size]   for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])   # targets shifted by 1
    return x.to(device), y.to(device)

for it in range(max_iters):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`estimate_loss()` periodically averages the loss over several batches on both splits, so the printed numbers aren't noisy single-batch estimates — useful for watching train vs. val loss diverge (or not) as training proceeds.

## Coding Goal: Text Generation & Sampling

Generation is autoregressive: feed in what's been generated so far, look only at the logits for the *last* position, turn them into a probability distribution, sample one character, append it, and repeat.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Two knobs control how "safe" vs. "adventurous" the output is:

- **Temperature** divides the logits before the softmax. `temperature < 1` sharpens the distribution toward the model's favorite next character (more repetitive, more confident); `temperature > 1` flattens it toward uniform (more random, sometimes incoherent).
- **Top-k** truncates the distribution to only the `k` most likely characters *before* sampling, zeroing out everything else. This prevents temperature from occasionally picking a wildly unlikely character from the long tail — a common failure mode of pure temperature sampling.

`gpt_train.py` demonstrates all three regimes back to back: `temperature=1.0` (no top-k), a focused `temperature=0.6, top_k=20`, and a looser `temperature=1.3, top_k=10`, so you can directly compare how the generated text changes.

---

## Try it yourself

```bash
python gpt_train.py
```

Because `article.txt` is small (~30KB), this tiny model will start reproducing chunks of the source text fairly quickly — a good way to *see* overfitting happen, and a nice segue into a later lesson on regularization and scaling up to larger, more diverse corpora.
