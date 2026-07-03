import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to Q and K tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Config:
    def __init__(self):
        self.block_size = 2048
        self.batch_size = 8
        self.learnin_rate = 3e-4
        self.max_steps = 500000 
        self.n_embd = 1024
        self.n_head = 16         # Number of Query Heads
        self.n_kv_head = 4       # Number of Key/Value Heads (GQA ratio is 16:4 -> 4 Qs share 1 KV)
        self.n_layer = 28
        self.dropout = 0.1
        self.head_size = self.n_embd // self.n_head
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_eval = 500

        


cfg = Config()

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute the cos and sin matrices for the max sequence length
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate the frequencies to apply to both halves of the vector
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len):
        # Return sliced cached matrices
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class GroupedQueryAttention(nn.Module):
    def __init__(self, window_size=None):
        super().__init__()
        self.num_q_heads = cfg.n_head
        self.num_kv_heads = cfg.n_kv_head
        self.head_dim = cfg.head_size
        self.num_rep = self.num_q_heads // self.num_kv_heads 
        self.window_size = window_size

        self.q_proj = nn.Linear(cfg.n_embd, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, cfg.n_embd, bias=False)

        # Q, k normalized to be consistent with the gemma architecture
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
      
        if window_size is not None:
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            mask = torch.triu(mask, diagonal=-window_size + 1)
            self.register_buffer('mask', mask.view(1, 1, cfg.block_size, cfg.block_size))
            
        self.dropout_p = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        # Project and reshape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # now we normalize
        q = self.q_norm(q)
        k = self.k_norm(k)


        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Repeat the KV heads
        k = torch.repeat_interleave(k, repeats=self.num_rep, dim=1)
        v = torch.repeat_interleave(v, repeats=self.num_rep, dim=1)

        # flash attention to make gpu go brrrr
        if self.window_size is None:
            # For global layers, PyTorch's native is_causal=True is hyper-optimized
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout_p if self.training else 0.0, 
                is_causal=True
            )
        else:
            # For Sliding Window layers, we pass our custom boolean mask
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=self.mask[:, :, :T, :T], 
                dropout_p=self.dropout_p if self.training else 0.0, 
                is_causal=False
            )
        
        # Reshape and project out
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_q_heads * self.head_dim)
        out = self.o_proj(out)
        
        # We manually apply dropout to the output projection
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        
        return out

class GemmaFeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.gate = nn.Linear(n_embd, 4 * n_embd, bias=False)
    self.up_proj = nn.Linear(n_embd, 4 * n_embd, bias=False)
    self.down_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
    self.act_fn = nn.GELU(approximate='tanh')
    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, x):
    gate = self.gate(x)
    up = self.up_proj(x)
    gate = self.act_fn(gate)
    x = gate * up
    x = self.down_proj(x)
    x = self.dropout(x)

    return x


class Block(nn.Module):
  def __init__(self, n_embd, n_head, window_size=None):

    super().__init__()
    self.sa = GroupedQueryAttention(window_size=window_size)
    # self.sa = MultiHeadAttention(n_head, head_size, window_size=window_size)
    # self.ffwd = FeedForward(n_embd)
    self.ffwd = GemmaFeedForward(n_embd)
    # self.ln1 = nn.LayerNorm(n_embd)
    # self.ln2 = nn.LayerNorm(n_embd)
    self.rms1 = nn.RMSNorm(n_embd)
    self.rms2 = nn.RMSNorm(n_embd)

  def forward(self, x, cos, sin):
        # x = x + self.sa(self.ln1(x))
        # x = x + self.ffwd(self.ln2(x))
        x = x + self.sa(self.rms1(x), cos, sin)
        x = x + self.ffwd(self.rms2(x))
        return x



class Gemma3LanguageModel(nn.Module, PyTorchModelHubMixin):

  def __init__(self, vocab_size):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
    # self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)

    self.rotary_emb = RotaryEmbedding(cfg.head_size, cfg.block_size)

    self.blocks = nn.ModuleList()

    for i in range(cfg.n_layer):
      # 1 % 6 mean layer 6th, 12th, 18th layers are global
      if i % 6 == 5:
        window_size = None
      else:
        window_size = 1024
      self.blocks.append(Block(cfg.n_embd, n_head=cfg.n_head, window_size=window_size))

    self.final_norm = nn.RMSNorm(cfg.n_embd)
    self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    # weight tying used to make the weights of embedding and
    # head point to the samme parameters it's easier to learn one representation 
    self.lm_head.weight = self.token_embedding_table.weight

    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, idx, targets=None):

    B, T = idx.shape # Corrected from B, T, C = idx.shape

    tok_emb = self.token_embedding_table(idx)
    # pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    
    # did this to match the gemma architecture
    tok_emb = tok_emb*(cfg.n_embd**0.5)
    
    # Fetch RoPE frequencies for the current sequence length
    cos, sin = self.rotary_emb(T)
    
    x = self.dropout(tok_emb) # Positional embeddings are no longer added here!
    
    for block in self.blocks:
        x = block(x, cos, sin)
        
    x = self.final_norm(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      # C here is vocab_size
      logits = logits.view(B*T, logits.shape[-1])
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
          # crop idx to the last block_size token
          idx_cond = idx[:, -cfg.block_size:]
          logits, loss = self(idx_cond)
          logits = logits[:, -1, :] # becomes (B, C)
          probs = F.softmax(logits, dim=-1) # (B, C)
          idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
          idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


@torch.no_grad()
def estimate_loss(model_name):
    out = {}
    if model_name == 'gemma':
      model = gemma_model
    else:
      model = model
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
