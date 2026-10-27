import random
from collections import deque

import torch
import torch.nn.functional as F


class ReplayBuffer:
    """FIFO store of student-generated samples, replayed by adaptive on-policy distillation."""

    def __init__(self, capacity, batch_size, pad_id):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.pad_values = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}

    def __len__(self):
        return len(self.memory)

    def add(self, model_batch, labels):
        batch = {"input_ids": model_batch["input_ids"].cpu(),
                 "attention_mask": model_batch["attention_mask"].cpu(),
                 "labels": labels.cpu()}
        for i in range(labels.size(0)):
            self.memory.append({k: v[i] for k, v in batch.items()})

    def sample(self, device):
        items = random.sample(self.memory, k=self.batch_size)
        # samples come from different batches, so pad them to a common length
        n = max(item["input_ids"].size(0) for item in items)
        out = {k: torch.stack([F.pad(item[k], (0, n - item[k].size(0)), value=v) for item in items]).to(device)
               for k, v in self.pad_values.items()}
        labels = out.pop("labels")
        return out, labels
