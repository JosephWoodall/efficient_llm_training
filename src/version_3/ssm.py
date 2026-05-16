import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMCell(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Continuous-time state transition matrices
        # In a real S4/Mamba, A is initialized analytically (e.g., HiPPO matrix)
        # It must be strictly negative to be stable
        self.A = nn.Parameter(torch.randn(d_model, d_state) - 5.0)
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dt_proj = nn.Parameter(torch.randn(d_model))
        
    def discretize(self):
        """
        Convert continuous-time parameters (A, B) to discrete-time (bar_A, bar_B)
        using zero-order hold (ZOH) approx.
        """
        dt = F.softplus(self.dt_proj).unsqueeze(-1) # (d_model, 1)
        
        # bar_A = exp(dt * A)
        bar_A = torch.exp(dt * self.A) # (d_model, d_state)
        
        # Simple discretization approximation for B: dt * B
        bar_B = dt * self.B # (d_model, d_state)
        
        return bar_A, bar_B
        
    def forward(self, x):
        """
        Parallel scan for fast GPU training.
        x shape: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model), final_hidden_state
        """
        batch, seq_len, _ = x.shape
        bar_A, bar_B = self.discretize()
        
        # Naive sequential scan for demonstration. 
        # A true parallel scan would use custom CUDA kernel (e.g. mamba_ssm)
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        y = []
        
        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(-1) # (batch, d_model, 1)
            h = bar_A * h + bar_B * x_t
            
            y_t = torch.sum(self.C * h, dim=-1) + self.D * x[:, t, :]
            y.append(y_t.unsqueeze(1))
            
        return torch.cat(y, dim=1), h

    def step(self, x_t, h_prev):
        """
        Iterative inference step O(1) memory.
        x_t: (batch, d_model)
        h_prev: (batch, d_model, d_state)
        """
        bar_A, bar_B = self.discretize()
        x_t_exp = x_t.unsqueeze(-1)
        h_t = bar_A * h_prev + bar_B * x_t_exp
        y_t = torch.sum(self.C * h_t, dim=-1) + self.D * x_t
        return y_t, h_t
