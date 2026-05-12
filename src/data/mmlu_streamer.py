import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader

class MMLUStreamer(IterableDataset):
    """
    Streams the MMLU dataset and formats it for multiple-choice evaluation.
    Format: "Question: <q> \n A) <ax> \n B) <bx> ... \n Answer:"
    """
    def __init__(self, tokenizer_name="gpt2", split="test", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # MMLU has many subsets; we'll stream a few common ones for a representative sample
        self.dataset = load_dataset("cais/mmlu", "all", split=split, streaming=True)
        self.max_length = max_length
        self.choices = ["A", "B", "C", "D"]

    def format_example(self, example):
        question = example["question"]
        options = example["choices"]
        answer_idx = example["answer"]
        
        prompt = f"Question: {question}\n"
        for i, opt in enumerate(options):
            prompt += f"{self.choices[i]}) {opt}\n"
        prompt += "Answer:"
        
        return prompt, answer_idx

    def __iter__(self):
        for example in self.dataset:
            prompt, answer_idx = self.format_example(example)
            
            # Tokenize prompt
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            yield {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "answer_idx": torch.tensor(answer_idx)
            }


def get_mmlu_loader(tokenizer_name="gpt2", batch_size=4, max_length=512):
    dataset = MMLUStreamer(tokenizer_name, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    print("Testing MMLU Streamer...")
    loader = get_mmlu_loader(batch_size=1)
    for i, batch in enumerate(loader):
        print(f"Example {i} input_ids shape: {batch['input_ids'].shape}")
        print(f"Answer IDX: {batch['answer_idx']}")
        if i >= 2: break
