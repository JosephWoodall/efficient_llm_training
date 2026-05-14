import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import json
import math
from src.arch.hybrid_mamba import HybridMambaMoE
from src.data.streamer import get_dataloader
from src.training.logger import setup_logger
from src.training.elf_eval import ELFEvaluator
from transformers import AutoTokenizer

# --- CONFIGURATION ---
METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_NAME = "ELF-Efficient-Production"
logger = setup_logger(LOG_NAME, METHOD_DIR)

# --- MODEL DEFINITION ---

class ELFModel(nn.Module):
    """
    ELF: Embedded Language Flows (arXiv:2605.10938)
    Enhanced with:
    1. Path-Energy Regularization (Optimal Transport)
    2. Self-Conditioning (Stability)
    3. Binary Decode Mode (Discrete mapping)
    4. SDE-inspired Sampler (Inference)
    """
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        
        # Continuous time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Mode embeddings: 0=[DENOISE], 1=[DECODE]
        self.mode_embed = nn.Embedding(2, d_model)
        
        # Self-conditioning projection
        self.self_cond_proj = nn.Linear(d_model, d_model)
        
        # Velocity prediction head
        self.v_head = nn.Linear(d_model, d_model)

    def forward(self, z_t, t, decode_mode=False, x_pred_cond=None):
        t_emb = self.time_embed(t).unsqueeze(1)
        
        mode_id = torch.ones(z_t.size(0), 1, dtype=torch.long, device=z_t.device) if decode_mode else torch.zeros(z_t.size(0), 1, dtype=torch.long, device=z_t.device)
        m_emb = self.mode_embed(mode_id)
        
        sc_emb = self.self_cond_proj(x_pred_cond) if x_pred_cond is not None else torch.zeros_like(z_t)
            
        h = z_t + t_emb + m_emb + sc_emb
        latents, _ = self.backbone(inputs_embeds=h, return_latents=True) 
        
        if decode_mode:
            return self.backbone.output(latents)
        return self.v_head(latents)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """SDE-inspired generation sampler."""
        self.eval()
        device = input_ids.device
        prompt_len = input_ids.shape[1]
        seq_len = prompt_len + max_new_tokens
        prompt_emb = self.backbone.embedding(input_ids)
        
        z_t = torch.randn(input_ids.shape[0], seq_len, self.d_model, device=device)
        z_t[:, :prompt_len, :] = prompt_emb
        
        x_pred_cond = torch.zeros_like(z_t)
        steps, dt = 32, 1.0 / 32
        
        for i in range(steps):
            t = torch.ones(input_ids.shape[0], 1, device=device) * (i * dt)
            v_pred = self.forward(z_t, t, decode_mode=False, x_pred_cond=x_pred_cond)
            x_pred_cond = (z_t + (1 - t.view(-1, 1, 1)) * v_pred).detach()
            
            # SDE noise injection
            sigma = 0.1 * (1 - t.item()) 
            z_t = z_t + dt * v_pred + math.sqrt(dt) * sigma * torch.randn_like(z_t)
            z_t[:, :prompt_len, :] = prompt_emb # Keep prompt fixed
            
        logits = self.forward(z_t, torch.ones(input_ids.shape[0], 1, device=device), decode_mode=True)
        return torch.argmax(logits, dim=-1)

# --- TRAINING LOOP ---

def run_production_training():
    device = torch.device("cpu")
    vocab_size, d_model = 50257, 128
    
    # Initialize Architecture
    backbone = HybridMambaMoE(vocab_size=vocab_size, d_model=d_model, n_layers=2)
    model = ELFModel(backbone, d_model).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loader = get_dataloader(batch_size=4)
    
    logger.info("Starting ELF Efficient Production Training (10,000 steps)...")
    
    max_steps = 10000
    for i, batch in enumerate(tqdm(loader, desc="ELF Training", total=max_steps)):
        ids = batch["input_ids"].to(device)
        with torch.no_grad(): x = backbone.embedding(ids)
            
        # Diffusion setup
        epsilon = torch.randn_like(x)
        t = torch.rand(x.size(0), 1, device=device)
        t_exp = t.view(-1, 1, 1)
        z_t = t_exp * x + (1 - t_exp) * epsilon
        target_v = x - epsilon
        
        # Self-Conditioning (50% prob)
        x_pred_cond = None
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                v_prev = model(z_t, t)
                x_pred_cond = (z_t + (1 - t_exp) * v_prev).detach()
        
        # Forward pass
        pred_v = model(z_t, t, x_pred_cond=x_pred_cond)
        
        # Loss: Flow Matching + Decode CE + Path Energy
        loss_flow = F.mse_loss(pred_v, target_v)
        loss_energy = torch.norm(pred_v, p=2, dim=-1).mean()
        
        # Decode mapping on clean embeddings
        logits = model(x, torch.ones_like(t), decode_mode=True)
        loss_ce = F.cross_entropy(logits.view(-1, vocab_size), ids.view(-1))
        
        # Combined Loss
        total_loss = loss_flow + 0.1 * loss_ce + 0.05 * loss_energy
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i > 0 and i % 100 == 0:
            logger.info(f"Step {i:05d} | Loss: {total_loss.item():.4f} | Flow: {loss_flow.item():.4f} | Energy: {loss_energy.item():.4f}")
        
        if i >= max_steps: break
        
    logger.info("Training complete. Saving results and running evaluation...")

    # --- EVALUATION ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    evaluator = ELFEvaluator(device=device)
    results = evaluator.evaluate(model, tokenizer, ["The future of AI is", "The meaning of life is"], num_samples=2)
    
    with open(os.path.join(METHOD_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Final Results: PPL={results['gen_ppl']:.2f}, Entropy={results['unigram_entropy']:.4f}")

if __name__ == "__main__":
    run_production_training()
