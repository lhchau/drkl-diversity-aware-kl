import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .mmap_dataset import MMapDataset


class LMTrainDataset(Dataset):
    """Prompt/response pairs tokenized by tools/process_data_dolly.py.

    Each item is `prompt + [SEP] + response`; collate() turns a batch into
    teacher-forcing inputs with the prompt masked out of the labels, plus the
    left-padded prompts used for generation.
    """

    def __init__(self, args, tokenizer, split, num=-1):
        self.data = MMapDataset(os.path.join(args.data_dir, f"{split}_0"))
        # the separator was written as -1, i.e. the max value of the storage dtype
        self.sep_id = np.iinfo(self.data.dtype).max
        self.pad_id = tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.num = len(self.data) if num == -1 else min(num, len(self.data))
        with open(os.path.join(args.data_dir, f"{split}.jsonl")) as f:
            outputs = [json.loads(line)["output"] for line in f]
        self.answers = [o if isinstance(o, list) else [o] for o in outputs]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        ids = self.data[index].astype(int)
        sep = np.where(ids == self.sep_id)[0][0]
        if sep > self.max_prompt_length:
            raise ValueError("prompt longer than --max-prompt-length: tokenize with the same value")
        return ids[:sep], np.concatenate([ids[:sep], ids[sep + 1:]])[:self.max_length]

    def collate(self, samples):
        bs = len(samples)
        length = max(2, max(len(ids) for _, ids in samples))
        model_batch = {
            "input_ids": torch.full((bs, length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, length),
        }
        labels = torch.full((bs, length), -100, dtype=torch.long)
        gen_batch = {
            "input_ids": torch.full((bs, self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        for i, (prompt, ids) in enumerate(samples):
            n = len(ids) - 1
            model_batch["input_ids"][i, :n] = torch.from_numpy(ids[:-1])
            model_batch["attention_mask"][i, :n] = 1.0
            labels[i, :n] = torch.from_numpy(ids[1:])
            labels[i, :len(prompt) - 1] = -100
            gen_batch["input_ids"][i, -len(prompt):] = torch.from_numpy(prompt)
            gen_batch["attention_mask"][i, -len(prompt):] = 1
        return model_batch, labels, gen_batch
