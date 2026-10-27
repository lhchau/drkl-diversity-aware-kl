"""Supervised fine-tuning (--loss sft) and knowledge distillation on Dolly.

After every epoch the model is evaluated on the validation split; the
checkpoint with the best ROUGE-L is kept in <save>/best_rougeL.
"""
import json
import os
import time

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from distillm import ReplayBuffer, SampleGenerator, distillation_loss, rkl_terms
from metrics import compute_metrics
from utils import (gather_list, get_dtype, get_model, get_tokenizer, initialize, load_causal_lm,
                   load_ds_config, print_rank, save_rank)

torch.set_num_threads(4)


def get_teacher_model(args, device, dtype):
    model = load_causal_lm(args.teacher_model_path, device, dtype)
    if args.teacher_peft_path is not None:
        model = PeftModel.from_pretrained(model, args.teacher_peft_path).merge_and_unload()
    return model.eval()


def get_optimizer(args, model):
    if args.peft:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    else:
        no_decay = ("bias", "ln_f.weight", "ln_1.weight", "ln_2.weight", "ln_cross_attn")
        params = list(model.named_parameters())
        groups = [
            {"params": [p for n, p in params if not any(nd in n for nd in no_decay)]},
            {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def lm_loss(logits, labels):
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))


def on_policy_batch(args, model, generator, buffer, model_batch, labels, gen_batch, threshold, progress, device):
    """DistiLLM's adaptive off-policy schedule: with probability decaying from `threshold`
    train on fresh student samples, otherwise (with probability `threshold`) replay old ones."""
    r = np.random.uniform(0, 1)
    if r < threshold * (1 - progress) or (r < threshold and len(buffer) < args.capacity):
        model_batch, labels = generator.run_sample(model, gen_batch)
        buffer.add(model_batch, labels)
    elif r < threshold and len(buffer) >= args.batch_size:
        model_batch, labels = buffer.sample(device)
    model.train()
    return model_batch, labels


@torch.no_grad()
def evaluate(args, tokenizer, model, dataset, device, epoch):
    """Validation loss and ROUGE-L of sampled responses."""
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers, collate_fn=dataset.collate)
    generation_config = GenerationConfig(do_sample=True, top_p=1.0, top_k=0, temperature=1.0,
                                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    model.eval()
    losses, responses = [], []
    for model_batch, labels, gen_batch in tqdm(loader, desc="validation", disable=dist.get_rank() != 0):
        model_batch, gen_batch, labels = to_device(model_batch, device), to_device(gen_batch, device), labels.to(device)
        logits = model(**model_batch, use_cache=False).logits
        losses.append(lm_loss(logits, labels).item())

        prompt_len = gen_batch["input_ids"].size(1)
        max_new_tokens = args.max_length - prompt_len
        if args.eval_max_new_tokens is not None:
            max_new_tokens = min(max_new_tokens, args.eval_max_new_tokens)
        output = model.generate(**gen_batch, generation_config=generation_config, max_new_tokens=max_new_tokens)
        responses += tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)

    val_loss = torch.tensor(np.mean(losses), device=device)
    dist.all_reduce(val_loss)
    val_loss = val_loss.item() / dist.get_world_size()
    responses = gather_list(responses)[:len(dataset)]
    rouge_l = compute_metrics(responses, dataset.answers[:len(dataset)])["rougeL"]

    if dist.get_rank() == 0:
        out_dir = os.path.join(args.save, "eval", str(epoch))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "answers.jsonl"), "w") as f:
            f.writelines(json.dumps({"text": r}) + "\n" for r in responses)
    return val_loss, rouge_l


def train(args, tokenizer, model, teacher, train_data, dev_data, device):
    sampler = DistributedSampler(train_data, shuffle=True, drop_last=True)
    loader = DataLoader(train_data, sampler=sampler, batch_size=args.batch_size, drop_last=True,
                        num_workers=args.num_workers, collate_fn=train_data.collate)
    generator = SampleGenerator(args, tokenizer) if args.on_policy else None
    buffer = ReplayBuffer(args.capacity, args.batch_size, tokenizer.eos_token_id) if args.on_policy else None
    threshold, prev_val_loss, best_rouge_l = args.init_threshold, float("inf"), float("-inf")
    log_path = os.path.join(args.save, "log.txt")
    running_loss, running_n, running_terms, tic = 0.0, 0, {}, time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for model_batch, labels, gen_batch in loader:
            model_batch, gen_batch, labels = to_device(model_batch, device), to_device(gen_batch, device), labels.to(device)
            progress = model.global_steps / args.total_iters
            if args.on_policy:
                model_batch, labels = on_policy_batch(args, model, generator, buffer, model_batch, labels, gen_batch,
                                                      threshold, progress, device)

            logits = model(**model_batch, use_cache=False).logits
            if teacher is None:
                loss = lm_loss(logits, labels)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(**model_batch, use_cache=False).logits
                loss = distillation_loss(args, logits, teacher_logits, labels, progress)
                if args.kd_ratio < 1:
                    loss = args.kd_ratio * loss + (1 - args.kd_ratio) * lm_loss(logits, labels)
                if args.log_rkl_terms:
                    terms = rkl_terms(logits, teacher_logits, labels)
                    values = torch.tensor(list(terms.values()), device=device)
                    dist.all_reduce(values)
                    for k, v in zip(terms, values.tolist()):
                        running_terms[k] = running_terms.get(k, 0.0) + v / dist.get_world_size()

            boundary = model.is_gradient_accumulation_boundary()
            model.backward(loss)
            model.step()

            dist.all_reduce(loss)
            running_loss += loss.item() / dist.get_world_size()
            running_n += 1
            if boundary and model.global_steps % args.log_interval == 0:
                log = (f"train | epoch {epoch} | step {model.global_steps}/{args.total_iters} | "
                       f"loss {running_loss / running_n:.4f} | lr {model.get_lr()[0]:.4e} | "
                       f"{(time.time() - tic) / args.log_interval:.2f}s/step")
                log += "".join(f" | {k} {v / running_n:.4f}" for k, v in running_terms.items())
                print_rank(log)
                save_rank(log, log_path)
                running_loss, running_n, running_terms, tic = 0.0, 0, {}, time.time()

        val_loss, rouge_l = evaluate(args, tokenizer, model, dev_data, device, epoch)
        log = f"valid | epoch {epoch} | loss {val_loss:.4f} | rougeL {rouge_l:.4f}"
        if args.on_policy:
            log += f" | threshold {threshold:.2f}"
        print_rank(log)
        save_rank(log, log_path)

        if rouge_l > best_rouge_l:
            best_rouge_l = rouge_l
            if dist.get_rank() == 0:
                path = os.path.join(args.save, "best_rougeL")
                tokenizer.save_pretrained(path)
                model.module.save_pretrained(path, safe_serialization=False)
                with open(os.path.join(path, "best_metric.json"), "w") as f:
                    json.dump({"epoch": epoch, "step": model.global_steps, "rougeL": rouge_l, "val_loss": val_loss}, f, indent=2)
                print_rank(f"new best ROUGE-L {rouge_l:.4f}, saved to {path}")
            dist.barrier()

        if args.on_policy:
            if val_loss >= prev_val_loss + args.loss_eps:
                threshold = min(threshold + 0.1, 1.0)
            prev_val_loss = val_loss
        tic = time.time()  # keep validation out of the s/step timing


def main():
    args = get_args()
    initialize(args)
    device = torch.cuda.current_device()

    ds_config = load_ds_config(args)
    ds_config.update(train_micro_batch_size_per_gpu=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                     gradient_clipping=args.clip_grad, steps_per_print=10 ** 7)
    args.deepspeed_config = None  # the config is passed to deepspeed.initialize directly
    dtype = get_dtype(ds_config)

    tokenizer = get_tokenizer(args)
    train_data = LMTrainDataset(args, tokenizer, "train")
    dev_data = LMTrainDataset(args, tokenizer, "valid", args.dev_num)
    steps_per_epoch = len(train_data) // dist.get_world_size() // args.batch_size
    args.total_iters = args.epochs * steps_per_epoch // args.gradient_accumulation_steps
    print_rank(f"train {len(train_data)} | valid {len(dev_data)} | optimizer steps {args.total_iters}")

    model = get_model(args, device, dtype, train=True)
    optimizer = get_optimizer(args, model)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args, lr_scheduler=lr_scheduler, config_params=ds_config)
    teacher = None if args.loss == "sft" else get_teacher_model(args, device, dtype)

    train(args, tokenizer, model, teacher, train_data, dev_data, device)


if __name__ == "__main__":
    main()
