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
logger = setup_logger("MoE-SSM", METHOD_DIR)

# --- ARCHITECTURE: MoE-SSM ---
# Each expert is a specialized SSM kernel.

class SSMExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Simplified SSM state: A, B, C matrices
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Very simplified recurrent update for demo
        # Real version would be a parallel scan
        gate = torch.sigmoid(self.A)
        return gate * self.B(x) + (1 - gate) * self.C(x)

class MoESSMBlock(nn.Module):
    def __init__(self, d_model, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([SSMExpert(d_model) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        
    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        
        # Mix expert outputs
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[:, :, i:i+1] * expert(x)
        return x + out

class MoESSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MoESSMBlock(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

def run_experiment():
    vocab_size = 50257
    model = MoESSMModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    logger.info("Research Idea: MoE-SSM (Routed Kernels) Training...")
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
