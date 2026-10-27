import torch
from transformers import GenerationConfig


class SampleGenerator:
    """Samples student responses to training prompts for on-policy distillation."""

    def __init__(self, args, tokenizer):
        self.max_new_tokens = args.max_length - args.max_prompt_length
        if args.student_gen_max_new_tokens is not None:
            self.max_new_tokens = min(self.max_new_tokens, args.student_gen_max_new_tokens)
        # prompts are left-padded with EOS (LMTrainDataset) and generate() pads with EOS too
        self.pad_id = tokenizer.eos_token_id
        self.generation_config = GenerationConfig(
            do_sample=True, top_p=1.0, top_k=0, temperature=1.0,
            eos_token_id=self.pad_id, pad_token_id=self.pad_id,
        )

    @torch.no_grad()
    def run_sample(self, model, gen_batch):
        """Returns right-padded prompt+response sequences and labels covering the response."""
        model.eval()
        prompt_len = gen_batch["input_ids"].size(1)
        full_ids = model.generate(**gen_batch, generation_config=self.generation_config,
                                  max_new_tokens=self.max_new_tokens)
        prompts = [ids[:prompt_len][ids[:prompt_len] != self.pad_id] for ids in full_ids]
        responses = [ids[prompt_len:][ids[prompt_len:] != self.pad_id] for ids in full_ids]
        width = max(len(p) + len(r) for p, r in zip(prompts, responses))
        input_ids = full_ids.new_full((len(prompts), width), self.pad_id)
        labels = full_ids.new_full((len(prompts), width), -100)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            n = len(prompt) + len(response)
            input_ids[i, :n] = torch.cat([prompt, response])
            # logits at position j predict token j + 1
            labels[i, len(prompt) - 1:n - 1] = response
        attention_mask = (input_ids != self.pad_id).float()
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels
