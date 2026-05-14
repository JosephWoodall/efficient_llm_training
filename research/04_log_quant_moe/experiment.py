import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from src.data.streamer import get_dataloader
from src.training.logger import setup_logger

# Set up logging to the current method directory
METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
logger = setup_logger("Log-Quant-MoE", METHOD_DIR)

# --- ARCHITECTURE: Log-Quant MoE ---
# Uses power-of-two weights to enable fast bit-shift operations on CPU.

def log_quant(w):
    # Quantize to {-4, -2, -1, 0, 1, 2, 4}
    sign = torch.sign(w)
    mag = torch.abs(w)
    # Map to nearest power of 2
    log_mag = torch.round(torch.log2(mag.clamp(min=1e-7)))
    log_mag = log_mag.clamp(-2, 2) # 2^-2 to 2^2
    quant = sign * torch.pow(2, log_mag)
    # Set values near zero to zero
    quant[mag < 0.1] = 0
    return w + (quant - w).detach()

class LogLinear(nn.Linear):
    def forward(self, x):
        w_quant = log_quant(self.weight)
        return F.linear(x, w_quant, self.bias)

class LogExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = LogLinear(d_model, d_ff)
        self.w2 = LogLinear(d_ff, d_model)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class LogMoEModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.experts = nn.ModuleList([LogExpert(d_model, d_model*2) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        logits = self.gate(x)
        expert_idx = torch.argmax(logits, dim=-1)
        
        # Simple routing for demonstration
        results = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                results[mask] = expert(x[mask])
        return self.output(results)

# --- TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = LogMoEModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    logger.info("Research Idea: Log-Quant MoE (Bit-Shift Friendly) Training...")
    for i, batch in enumerate(tqdm(loader)):
        ids = batch["input_ids"]
        out = model(ids[:, :-1])
        loss = criterion(out.reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= 10: break
    logger.info(f"Finished. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_experiment()
