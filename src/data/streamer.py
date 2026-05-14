import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader

class TokenizedStreamer(IterableDataset):
    def __init__(self, tokenizer_name="gpt2", split="train", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Use an instruction tuning dataset for advanced reasoning
        self.dataset = load_dataset("tatsu-lab/alpaca", split=split, streaming=True)
        self.max_length = max_length

    def __iter__(self):
        for example in self.dataset:
            text = example.get("text", "")
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
    print("Testing instruction data streamer...")
    loader = get_dataloader(batch_size=2)
    for i, batch in enumerate(loader):
        print(f"Batch {i}: input_ids shape {batch['input_ids'].shape}")
        if i >= 2: break
