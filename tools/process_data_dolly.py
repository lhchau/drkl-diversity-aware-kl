"""Tokenize Dolly into the binary train/valid splits read by finetune.py.

    python tools/process_data_dolly.py --data-dir data/dolly --save processed_data/dolly/full \
        --model-path checkpoints/gpt2-base --model-type gpt2

writes <save>/<model-type>/{train,valid}_0.{bin,idx} and {train,valid}.jsonl,
holding out the first --dev-num examples of raw.jsonl for validation.
With --eval-sets, it instead renders the prompts of each benchmark's valid.jsonl
into valid_<model-type>.jsonl (needed for chat-template models such as Qwen3).
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.mmap_dataset import MMapDatasetBuilder  # noqa: E402

PROMPT = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Response:\n")
PROMPT_WITH_INPUT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                     "Write a response that appropriately completes the request.\n\n"
                     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
# models trained through their chat template; the others (all base models) use the plain prompt above
CHAT_TEMPLATE_MODELS = ("qwen3",)


class Encoder:
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def render(self, line):
        """Returns the prompt string and, for chat-template models, the chat messages."""
        template = PROMPT_WITH_INPUT if line.get("input") else PROMPT
        prompt = template.format(instruction=line["instruction"], input=line.get("input", ""))
        if self.args.model_type not in CHAT_TEMPLATE_MODELS:
            return prompt, None
        messages = [{"role": "user", "content": prompt.removesuffix("\n\n### Response:\n")}]
        prompt = Encoder.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return prompt, messages

    def encode(self, raw):
        line = json.loads(raw)
        tok = Encoder.tokenizer
        prompt, messages = self.render(line)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if messages is None:
            full_ids = tok.encode(prompt + line["output"], add_special_tokens=False) + [tok.eos_token_id]
        else:
            full = tok.apply_chat_template(messages + [{"role": "assistant", "content": line["output"]}],
                                           tokenize=False, add_generation_prompt=False, enable_thinking=False)
            full_ids = tok.encode(full, add_special_tokens=False)
        return line, prompt, prompt_ids[:self.args.max_prompt_length], full_ids[len(prompt_ids):]


def process_dolly(args, encoder):
    out_dir = os.path.join(args.save, args.model_type)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw = f.readlines()
    assert args.dev_num > 0, "--dev-num must hold out at least one validation example"
    splits = {"valid": raw[:args.dev_num], "train": raw[args.dev_num:]}
    # vocabularies above 65535 tokens (Llama-3, Qwen3) need 32-bit ids
    dtype = np.uint16 if args.model_type in ("gpt2", "opt") else np.uint32

    with multiprocessing.Pool(args.workers, initializer=encoder.initializer) as pool:
        for split, lines in splits.items():
            builder = MMapDatasetBuilder(os.path.join(out_dir, f"{split}_0"), dtype)
            prompt_lens, response_lens = [], []
            with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
                for line, prompt, prompt_ids, response_ids in pool.imap(encoder.encode, lines, chunksize=50):
                    builder.add_item(prompt_ids + [-1] + response_ids)  # -1 separates prompt and response
                    f.write(json.dumps({"instruction": line["instruction"], "prompt": prompt,
                                        "input": line.get("input", ""), "output": line["output"]}) + "\n")
                    prompt_lens.append(len(prompt_ids))
                    response_lens.append(len(response_ids))
            builder.finalize()
            print(f"{split}: {len(prompt_lens)} examples | prompt tokens mean {np.mean(prompt_lens):.1f} "
                  f"max {max(prompt_lens)} | response tokens mean {np.mean(response_lens):.1f} max {max(response_lens)}")


def render_eval_sets(args, encoder):
    encoder.initializer()
    for data_dir in args.eval_sets:
        with open(os.path.join(data_dir, "valid.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        path = os.path.join(data_dir, f"valid_{args.model_type}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps({**row, "prompt": encoder.render(row)[0]}) + "\n")
        print(f"wrote {len(rows)} prompts to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="any checkpoint with the target tokenizer")
    parser.add_argument("--model-type", required=True, choices=["gpt2", "opt", "llama3", "qwen3"])
    parser.add_argument("--data-dir", default="data/dolly")
    parser.add_argument("--save", default="processed_data/dolly/full")
    parser.add_argument("--dev-num", type=int, default=1000)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--eval-sets", nargs="+", help="benchmark directories whose valid.jsonl prompts to render")
    args = parser.parse_args()

    encoder = Encoder(args)
    if args.eval_sets:
        render_eval_sets(args, encoder)
    else:
        process_dolly(args, encoder)


if __name__ == "__main__":
    main()
