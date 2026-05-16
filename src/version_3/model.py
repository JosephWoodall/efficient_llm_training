import torch
import torch.nn as nn
from src.version_3.tssm_block import TSSMBlock
from src.version_3.tokenizer import ASTTokenizer

class SyntaxGatedHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, x, allowed_tokens_mask=None):
        """
        x: (batch, seq_len, d_model) or (batch, d_model)
        allowed_tokens_mask: boolean mask of shape (batch, vocab_size), 
                             where True means allowed, False means forbidden.
        """
        logits = self.proj(x)
        
        if allowed_tokens_mask is not None:
            # Set forbidden tokens to -inf
            if logits.dim() == 3:
                logits = logits.masked_fill(~allowed_tokens_mask, float('-inf'))
            else:
                logits = logits.masked_fill(~allowed_tokens_mask, float('-inf'))
                
        return logits

class TSSM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TSSMBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = SyntaxGatedHead(d_model, vocab_size)
        self.vocab_size = vocab_size
        
        # Initialize token tracking
        tokenizer = ASTTokenizer()
        self.id_to_syntax = tokenizer.id_to_syntax
        self.syntax_vocab = tokenizer.syntax_vocab
        self._build_syntax_transitions()

    def _build_syntax_transitions(self):
        # A mock transition table for AST states.
        # e.g., 'def' -> 'identifier' -> 'parameters' -> 'block' -> 'return'
        self.allowed_transitions = {
            self.syntax_vocab["UNK"]: list(self.syntax_vocab.values()),
            self.syntax_vocab["PAD"]: [self.syntax_vocab["PAD"]],
            self.syntax_vocab["def"]: [self.syntax_vocab["identifier"]],
            self.syntax_vocab["identifier"]: [
                self.syntax_vocab["parameters"], 
                self.syntax_vocab["assignment"],
                self.syntax_vocab["call"]
            ],
            self.syntax_vocab["parameters"]: [self.syntax_vocab["block"]],
            self.syntax_vocab["block"]: [
                self.syntax_vocab["return_statement"],
                self.syntax_vocab["expression_statement"],
                self.syntax_vocab["if_statement"]
            ],
        }

    def get_allowed_mask(self, last_token_ids):
        """
        last_token_ids: (batch,)
        Returns: (batch, vocab_size) boolean mask
        """
        batch_size = last_token_ids.shape[0]
        mask = torch.zeros(batch_size, self.vocab_size, dtype=torch.bool, device=last_token_ids.device)
        
        for i in range(batch_size):
            token_id = last_token_ids[i].item()
            allowed = self.allowed_transitions.get(token_id, list(self.syntax_vocab.values()))
            mask[i, allowed] = True
            
        return mask

    def forward(self, x, apply_syntax_mask=False):
        """
        x: (batch, seq_len)
        """
        hidden = self.embedding(x)
        
        for layer in self.layers:
            hidden, _ = layer(hidden)
            
        hidden = self.norm_f(hidden)
        
        mask = None
        if apply_syntax_mask:
            last_tokens = x[:, -1]
            mask = self.get_allowed_mask(last_tokens)
            if hidden.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, hidden.size(1), -1)
                
        logits = self.head(hidden, allowed_tokens_mask=mask)
        return logits
        
    def generate(self, start_tokens, max_length=50):
        # simple step-by-step inference
        current_tokens = start_tokens
        
        for _ in range(max_length):
            logits = self.forward(current_tokens, apply_syntax_mask=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
        return current_tokens
