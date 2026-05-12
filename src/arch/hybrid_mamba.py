import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class SimpleMambaBlock(nn.Module):
    # O(N) simplified SSM
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        res = x
        x = self.in_proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        
        x1 = x1.transpose(1, 2)
        # Causal padding: pad left by kernel_size - 1 (which is 2)
        x1 = F.pad(x1, (2, 0))
        x1 = self.conv(x1).transpose(1, 2)
        x1 = F.silu(x1)
        
        gate = torch.sigmoid(self.dt_proj(x1))
        ssm_out = x1 * gate + (1 - gate) * x2
        
        out = self.out_proj(ssm_out * F.silu(x2))
        return out + res

class MoEExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 4)
        self.w2 = nn.Linear(d_model * 4, d_model)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class SharedExpertMoE(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.shared_expert = MoEExpert(d_model)
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([MoEExpert(d_model) for _ in range(n_experts)])
        
    def forward(self, x):
        # Shared expert processes all tokens
        shared_out = self.shared_expert(x)
        
        # Routed experts
        batch, seq, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        gate_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(F.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_output = expert(x_flat[mask])
                for k in range(self.top_k):
                    k_mask = (selected_experts[mask, k] == i)
                    if k_mask.any():
                        routed_out[mask.nonzero().squeeze(1)[k_mask]] += weights[mask, k][k_mask].unsqueeze(1) * expert_output[k_mask]
        
        return shared_out + routed_out.view(batch, seq, d_model), gate_logits

class HybridBlock(nn.Module):
    def __init__(self, d_model, num_heads, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = nn.LayerNorm(d_model)
        if use_attention:
            self.seq_mixer = SelfAttention(d_model, num_heads)
        else:
            self.seq_mixer = SimpleMambaBlock(d_model)
            
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SharedExpertMoE(d_model)

    def forward(self, x):
        x = x + self.seq_mixer(self.norm1(x))
        moe_out, gate_logits = self.moe(self.norm2(x))
        x = x + moe_out
        return x, gate_logits

class HybridMambaMoE(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, num_heads=8, attn_every=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridBlock(d_model, num_heads, use_attention=(i % attn_every == 0))
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        all_gate_logits = []
        for layer in self.layers:
            x, gate_logits = layer(x)
            all_gate_logits.append(gate_logits)
            
        # Calculate load balancing loss
        aux_loss = 0.0
        for gate_logits in all_gate_logits:
            probs = F.softmax(gate_logits, dim=-1)
            # fraction of tokens routed to each expert
            f = probs.mean(dim=0)
            # if perfectly balanced, f = 1/N. Minimizing f.pow(2).sum() encourages balance
            aux_loss += f.pow(2).sum() * len(f)
            
        return self.output(self.norm(x)), aux_loss

if __name__ == "__main__":
    model = HybridMambaMoE(vocab_size=1000, d_model=128, n_layers=4)
    out, aux = model(torch.randint(0, 1000, (1, 16)))
    print("Success! Output shape:", out.shape)
    print(f"Aux Loss: {aux.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
