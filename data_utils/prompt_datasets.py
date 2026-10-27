import json
import os

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Evaluation prompts and reference answers of one benchmark.

    Reads `<data_dir>/valid_<model_type>.jsonl` when it exists (prompts rendered
    with a chat template by tools/process_data_dolly.py --eval-sets), otherwise
    the benchmark's own `valid.jsonl`.
    """

    def __init__(self, args, tokenizer, data_dir):
        path = os.path.join(data_dir, f"valid_{args.model_type}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(data_dir, "valid.jsonl")
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        # chat-template prompts already contain their special tokens
        add_special_tokens = args.model_type != "qwen3"
        self.prompts = [
            tokenizer.encode(r["prompt"].replace("<n>", "\n"), add_special_tokens=add_special_tokens)[:args.max_prompt_length]
            for r in rows
        ]
        self.answers = [r["output"] if isinstance(r["output"], list) else [r["output"]] for r in rows]
        # the first reference, teacher-forced to measure the model's entropy and confidence
        self.answer_ids = [tokenizer.encode(a[0], add_special_tokens=False) + [tokenizer.eos_token_id] for a in self.answers]
        self.pad_id = tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        return prompt, self.answer_ids[index][:self.max_length - len(prompt)]

    def collate(self, samples):
        """Left-padded prompts for generation, and right-padded prompt + answer with labels on the answer."""
        bs = len(samples)
        gen_batch = {
            "input_ids": torch.full((bs, self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        length = max(len(prompt) + len(answer) for prompt, answer in samples)
        answer_batch = {
            "input_ids": torch.full((bs, length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, length, dtype=torch.long),
        }
        labels = torch.full((bs, length), -100, dtype=torch.long)
        for i, (prompt, answer) in enumerate(samples):
            gen_batch["input_ids"][i, -len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_batch["attention_mask"][i, -len(prompt):] = 1
            n = len(prompt) + len(answer)
            answer_batch["input_ids"][i, :n] = torch.tensor(prompt + answer, dtype=torch.long)
            answer_batch["attention_mask"][i, :n] = 1
            labels[i, len(prompt) - 1:n - 1] = torch.tensor(answer, dtype=torch.long)  # logits at j predict token j + 1
        return gen_batch, answer_batch, labels
