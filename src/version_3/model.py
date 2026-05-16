import torch
import torch.nn as nn
from src.version_3.tssm_block import TSSMBlock
from src.version_3.tokenizer import HybridTokenizer

class DualHead(nn.Module):
    def __init__(self, d_model, vocab_size, syntax_vocab_size):
        super().__init__()
        # The Language Head predicts standard BPE text tokens
        self.language_head = nn.Linear(d_model, vocab_size, bias=False)
        # The Syntax Head predicts structural logic for AST gating
        self.syntax_head = nn.Linear(d_model, syntax_vocab_size, bias=False)

    def forward(self, x):
        lang_logits = self.language_head(x)
        syntax_logits = self.syntax_head(x)
        return lang_logits, syntax_logits

class HybridTSSM(nn.Module):
    def __init__(self, vocab_size, syntax_vocab_size, d_model=128, n_layers=2, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TSSMBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = DualHead(d_model, vocab_size, syntax_vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        hidden = self.embedding(x)
        
        for layer in self.layers:
            hidden, _ = layer(hidden)
            
        hidden = self.norm_f(hidden)
        lang_logits, syntax_logits = self.head(hidden)
        return lang_logits, syntax_logits
        
    def generate(self, start_tokens, max_length=50, tokenizer=None):
        current_tokens = start_tokens
        
        # State tracking for Dynamic Gating
        in_code_block = False
        if tokenizer:
            generated_text = tokenizer.decode_text(start_tokens[0].cpu().tolist())
            if "```python" in generated_text and not generated_text.endswith("```"):
                in_code_block = True
        
        for _ in range(max_length):
            lang_logits, syntax_logits = self.forward(current_tokens)
            next_token_logits = lang_logits[:, -1, :]
            
            # Simulated Dynamic Gating: If inside a code block, the Syntax Head 
            # determines if the predicted text token breaks the AST.
            if in_code_block and tokenizer:
                # We would run `tree_sitter` here on the top-k proposed tokens.
                # For invalid parses, we set logit to -inf (Syntax Gate closes).
                pass
                
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            if tokenizer:
                new_text = tokenizer.decode_text([next_token.item()])
                if "```python" in new_text:
                    in_code_block = True
                elif "```" in new_text and in_code_block:
                    in_code_block = False
            
        return current_tokens
