import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.data.streamer import get_dataloader

# --- ARCHITECTURE: Ternary Linear Attention ---
# $O(N)$ attention with BitNet b1.58 weights.

def ternary_quant(w):
    scale = w.abs().mean() + 1e-7
    w_quant = torch.round(w / scale).clamp(-1, 1)
    return w + (w_quant - w).detach()

class TernaryLinear(nn.Linear):
    def forward(self, x):
        w = ternary_quant(self.weight)
        return F.linear(x, w, self.bias)

class TernaryLinearAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)
        
    def forward(self, x):
        # x: [batch, seq, d_model]
        q = F.elu(self.q_proj(x)) + 1
        k = F.elu(self.k_proj(x)) + 1
        v = self.v_proj(x)
        
        # Linear Attention: (Q @ (K.T @ V)) / (Q @ K.T.sum)
        # We compute this sequentially for O(1) memory during inference
        # For training, we use the parallel form
        kv = torch.matmul(k.transpose(-2, -1), v)
        z = 1 / (torch.matmul(q, k.transpose(-2, -1).sum(dim=-1, keepdim=True)) + 1e-7)
        
        out = torch.matmul(q, kv) * z
        return self.out_proj(out)

class TernaryLAModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TernaryLinearAttention(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)

def run_experiment():
    vocab_size = 50257
    model = TernaryLAModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    print("Research Idea: Ternary Linear Attention Training...")
    for i, batch in enumerate(tqdm(loader)):
        ids = batch["input_ids"]
        out = model(ids[:, :-1])
        loss = criterion(out.reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= 10: break
    print(f"Finished. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_experiment()
