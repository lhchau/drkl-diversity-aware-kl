import json
import os
import random
from datetime import timedelta

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_rank(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def save_rank(log_str, path):
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(path, "a") as f:
            f.write(log_str + "\n")


def initialize(args):
    """Set up the process group, a per-rank seed and the output directory."""
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed(timeout=timedelta(minutes=300))
    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dist.get_rank() == 0:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    dist.barrier()


def gather_list(items):
    """all_gather a per-rank list of a DistributedSampler(shuffle=False) and restore dataset order."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, items)
    # rank r holds indices r, r + W, r + 2W, ...
    return [x for group in zip(*gathered) for x in group]


def load_ds_config(args):
    with open(args.deepspeed_config) as f:
        return json.load(f)


def get_dtype(ds_config):
    if ds_config.get("bf16", {}).get("enabled", False):
        return torch.bfloat16
    if ds_config.get("fp16", {}).get("enabled", False):
        return torch.float16
    return torch.float32


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ("gpt2", "opt") or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_causal_lm(path, device, dtype):
    """Prefer flash-attention-2, then SDPA, then the default attention implementation."""
    for attn in ("flash_attention_2", "sdpa", None):
        try:
            return AutoModelForCausalLM.from_pretrained(
                path, device_map={"": device}, torch_dtype=dtype, attn_implementation=attn)
        except Exception:
            if attn is None:
                raise


def get_model(args, device, dtype, train):
    model = load_causal_lm(args.model_path, device, dtype)
    if args.peft == "lora":
        if args.peft_path is not None and not train:
            model = PeftModel.from_pretrained(model, args.peft_path)
        else:
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=not train, r=args.peft_lora_r,
                lora_alpha=args.peft_lora_alpha, lora_dropout=args.peft_lora_dropout))
        if dist.get_rank() == 0:
            model.print_trainable_parameters()
    else:
        print_rank(f" > number of parameters: {sum(p.nelement() for p in model.parameters())}")
    if args.gradient_checkpointing and train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model
