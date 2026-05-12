import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.data.streamer import get_dataloader

# --- ARCHITECTURE: Hyper-MoE-SSM ---
# Combines weight generation with routed SSM experts for maximum efficiency.

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

class HyperMoEExpert(nn.Module):
    def __init__(self, d_model, seed_dim):
        super().__init__()
        self.w_gen = WeightGenerator(seed_dim, (d_model, d_model))
        self.seed = nn.Parameter(torch.randn(seed_dim))
        
    def forward(self, x):
        w = self.w_gen(self.seed)
        return F.linear(x, w)

class HyperMoESSMBlock(nn.Module):
    def __init__(self, d_model, seed_dim, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([HyperMoEExpert(d_model, seed_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        
    def forward(self, x):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[:, :, i:i+1] * expert(x)
        return x + out

class HyperMoESSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, seed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([HyperMoESSMBlock(d_model, seed_dim) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

def run_experiment():
    vocab_size = 50257
    model = HyperMoESSMModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    print("Research Idea: Hyper-MoE-SSM (Ultimate Hybrid) Training...")
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
