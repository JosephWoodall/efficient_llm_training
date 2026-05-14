import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader

class MMLUStreamer(IterableDataset):
    """
    Streams the MMLU dataset and formats it for multiple-choice evaluation using 5-shot prompts.
    """
    def __init__(self, tokenizer_name="gpt2", split="test", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # MMLU has many subsets; we'll stream a few common ones for a representative sample
        self.dataset = load_dataset("cais/mmlu", "all", split=split, streaming=True)
        self.max_length = max_length
        self.choices = ["A", "B", "C", "D"]
        
        # Fetch 5 shots from the dev split
        ds_dev = load_dataset("cais/mmlu", "all", split="dev", streaming=True)
        self.few_shot_examples = ""
        for i, ex in enumerate(ds_dev):
            self.few_shot_examples += f"Question: {ex['question']}\n"
            for j, opt in enumerate(ex['choices']):
                self.few_shot_examples += f"{self.choices[j]}) {opt}\n"
            self.few_shot_examples += f"Answer: {self.choices[ex['answer']]}\n\n"
            if i >= 4: break

    def format_example(self, example):
        question = example["question"]
        options = example["choices"]
        answer_idx = example["answer"]
        
        instruction = f"Answer the following multiple choice question by providing only the letter of the correct choice. Here are some examples:\n\n{self.few_shot_examples}Now, answer this question:\nQuestion: {question}\n"
        for i, opt in enumerate(options):
            instruction += f"{self.choices[i]}) {opt}\n"
            
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n### Response:\n"
        
        return prompt, answer_idx

    def __iter__(self):
        for example in self.dataset:
            prompt, answer_idx = self.format_example(example)
            
            # Tokenize prompt, truncate from left so we never cut off the actual question
            # However, HF tokenizer pad/truncate is weird. We can manually truncate tokens if needed.
            # `truncation=True` defaults to right truncation. 
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


def get_mmlu_loader(tokenizer_name="gpt2", batch_size=4, max_length=2048):
    dataset = MMLUStreamer(tokenizer_name, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    print("Testing MMLU Streamer...")
    loader = get_mmlu_loader(batch_size=1)
    for i, batch in enumerate(loader):
        print(f"Example {i} input_ids shape: {batch['input_ids'].shape}")
        print(f"Answer IDX: {batch['answer_idx']}")
        if i >= 2: break
