import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader

class TokenizedStreamer(IterableDataset):
    def __init__(self, tokenizer_name="gpt2", split="train", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Mix web text (grammar/reasoning) with Wikipedia (facts)
        ds_web = load_dataset("Skylion007/openwebtext", split=split, streaming=True)
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split=split, streaming=True)
        
        # Interleave: roughly 50% web, 50% wiki
        self.dataset = interleave_datasets([ds_web, ds_wiki], probabilities=[0.5, 0.5])
        self.max_length = max_length

    def __iter__(self):
        for example in self.dataset:
            text = example.get("text", example.get("content", ""))
            if not text:
                continue
                
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            yield {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }

def get_dataloader(tokenizer_name="gpt2", batch_size=4, max_length=512):
    dataset = TokenizedStreamer(tokenizer_name, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    print("Testing mixed data streamer...")
    loader = get_dataloader(batch_size=2)
    for i, batch in enumerate(loader):
        print(f"Batch {i}: input_ids shape {batch['input_ids'].shape}")
        if i >= 2: break
