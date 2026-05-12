import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.data.streamer import get_dataloader

# --- ARCHITECTURE: Hyper-Diff-SSM ---
# Combines Hyper-BitNet (Weight Generation) with Diff-SSM (Global Memory).

class WeightGenerator(nn.Module):
    def __init__(self, seed_dim, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(seed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, torch.prod(torch.tensor(out_shape)).item())
        )
    def forward(self, seed):
        w = self.net(seed).view(self.out_shape)
        scale = w.abs().mean() + 1e-7
        w_quant = torch.round(w / scale).clamp(-1, 1)
        return w + (w_quant - w).detach()

class HyperDiffBlock(nn.Module):
    def __init__(self, d_model, seed_dim):
        super().__init__()
        self.d_model = d_model
        self.global_memory = nn.Parameter(torch.zeros(1, d_model))
        # Generated weights for projection
        self.w_gen = WeightGenerator(seed_dim, (d_model * 2, d_model))
        self.seed = nn.Parameter(torch.randn(seed_dim))
        self.surprise_gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        res = x
        # Use generated ternary weights for input projection
        w = self.w_gen(self.seed)
        x_proj = F.linear(x, w)
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        surprise = torch.sigmoid(self.surprise_gate(x1))
        context = x1 * (1 - surprise) + self.global_memory * surprise
        
        out = self.out_proj(context * F.silu(x2))
        return out + res

class HyperDiffModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, seed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([HyperDiffBlock(d_model, seed_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

def run_experiment():
    vocab_size = 50257
    model = HyperDiffModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    print("Research Idea: Hyper-Diff-SSM (Hybrid) Training...")
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
