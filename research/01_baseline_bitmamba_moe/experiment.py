import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.streamer import get_dataloader

# --- ARCHITECTURE ---

def weight_quant(w):
    scale = w.abs().mean()
    e = 1e-7
    quant = torch.round(w / (scale + e)).clamp(-1, 1)
    return w + (quant - w).detach()

class BitLinear(nn.Linear):
    def forward(self, x):
        w_quant = weight_quant(self.weight)
        x_norm = F.layer_norm(x, (x.shape[-1],))
        x_quant = x_norm + (torch.round(x_norm * 127).clamp(-128, 127) / 127 - x_norm).detach()
        return F.linear(x_quant, w_quant, self.bias)

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = BitLinear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dt_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)
        
    def forward(self, x):
        res = x
        x = self.in_proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        x1 = x1.transpose(1, 2)
        x1 = self.conv(x1).transpose(1, 2)
        x1 = F.silu(x1)
        gate = torch.sigmoid(self.dt_proj(x1))
        ssm_out = x1 * gate + (1 - gate) * x2
        out = self.out_proj(ssm_out * F.silu(x2))
        return out + res

class MoEExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class BitMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([MoEExpert(d_model, d_ff) for _ in range(num_experts)])
        
    def forward(self, x):
        batch, seq, d_model = x.shape
        x_flat = x.view(-1, d_model)
        gate_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(F.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        results = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_output = expert(x_flat[mask])
                for k in range(self.top_k):
                    k_mask = (selected_experts[mask, k] == i)
                    if k_mask.any():
                        results[mask.nonzero().squeeze(1)[k_mask]] += weights[mask, k][k_mask].unsqueeze(1) * expert_output[k_mask]
        return results.view(batch, seq, d_model)

class BitMambaMoE(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "ssm": SimpleMambaBlock(d_model),
            "moe": BitMoE(d_model, d_model * 4, num_experts=n_experts)
        }) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = BitLinear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer["ssm"](x)
            x = x + layer["moe"](x)
        return self.output(self.norm(x))

# --- TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = BitMambaMoE(vocab_size=vocab_size)
    device = torch.device("cpu")
    model.to(device)
    
    loader = get_dataloader(batch_size=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Baseline BitMamba-MoE Training...")
    for i, batch in enumerate(tqdm(loader)):
        ids = batch["input_ids"].to(device)
        out = model(ids[:, :-1])
        loss = criterion(out.reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i >= 10: break
    print(f"Finished. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_experiment()
