import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from tqdm import tqdm

class ELFEvaluator:
    """
    Evaluator based on 'ELF: Embedded Language Flows' (arXiv:2605.10938).
    Focuses on Quality (Generative Perplexity) and Diversity (Unigram Entropy).
    """
    def __init__(self, evaluator_model_name="gpt2-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading ELF evaluator model: {evaluator_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(evaluator_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(evaluator_model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def calculate_gen_ppl(self, generated_texts):
        """
        Calculates Generative Perplexity (Gen. PPL) as described in the ELF paper.
        Measures the fluency of generated samples using a pretrained GPT-2 Large model.
        """
        if not generated_texts:
            return float('inf')
            
        total_loss = 0
        total_tokens = 0
        
        for text in tqdm(generated_texts, desc="Calculating Gen. PPL"):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            if input_ids.size(1) <= 1:
                continue
                
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.size(1)
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
        if total_tokens == 0:
            return float('inf')
            
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def calculate_unigram_entropy(self, generated_token_ids, vocab_size):
        """
        Calculates Average Unigram Entropy (H) to measure diversity.
        Higher entropy indicates better diversity in the generated vocabulary.
        """
        if not generated_token_ids:
            return 0.0
            
        # Flatten token list if it's a list of lists
        if isinstance(generated_token_ids[0], list):
            tokens = [t for sublist in generated_token_ids for t in sublist]
        else:
            tokens = generated_token_ids
            
        tokens_tensor = torch.tensor(tokens)
        counts = torch.bincount(tokens_tensor, minlength=vocab_size)
        probs = counts.float() / counts.sum()
        probs = probs[probs > 0]
        
        entropy = -torch.sum(probs * torch.log(probs)).item()
        return entropy

    def evaluate(self, model, tokenizer, prompts, num_samples=100, max_new_tokens=50):
        """
        Full evaluation flow: Generation -> Metrics.
        """
        model.eval()
        generated_texts = []
        all_token_ids = []
        
        print(f"Generating {num_samples} samples for ELF evaluation...")
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else 'cpu')
            
            with torch.no_grad():
                # Simple greedy/top-p sampling for evaluation
                # In ELF paper, they use specific continuous flow sampling, 
                # but for AR models we use standard generation.
                output_ids = model.generate(
                    input_ids, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
            all_token_ids.extend(output_ids[0].tolist())
            
        gen_ppl = self.calculate_gen_ppl(generated_texts)
        entropy = self.calculate_unigram_entropy(all_token_ids, tokenizer.vocab_size)
        
        return {
            "gen_ppl": gen_ppl,
            "unigram_entropy": entropy,
            "num_samples": num_samples
        }

# --- ELF Continuous Flow Sampler (Template) ---

def elf_flow_sampler(model, z0, steps=64, dt=1/64):
    """
    Pseudocode implementation of the ELF sampling flow from the paper.
    Requires a model that predicts the clean embedding x_pred from z_t.
    """
    z_t = z0
    for i in range(steps):
        t = i * dt
        # Predict clean embedding x_pred
        x_pred = model.predict_clean(z_t, t) 
        
        # Calculate velocity v = (x_pred - z_t) / (1 - t)
        velocity = (x_pred - z_t) / (1 - t + 1e-6)
        
        # Update z_{t+dt} = z_t + dt * v
        z_t = z_t + dt * velocity
        
    # Final step: decode z_1
    tokens = model.decode(z_t)
    return tokens
