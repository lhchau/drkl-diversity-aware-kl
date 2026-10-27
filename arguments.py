import argparse
import os

import deepspeed

from distillm.losses import LOSSES


def get_args():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("model")
    g.add_argument("--model-path", required=True, help="student (or SFT) model: HF hub id or local path")
    g.add_argument("--model-type", required=True, choices=["gpt2", "opt", "llama3", "qwen3"])
    g.add_argument("--teacher-model-path", help="teacher model; required unless --loss sft")
    g.add_argument("--teacher-peft-path", help="LoRA adapter merged into the teacher")
    g.add_argument("--peft", choices=["lora"], help="train / evaluate the model as a LoRA adapter")
    g.add_argument("--peft-path", help="LoRA adapter to evaluate")
    g.add_argument("--peft-lora-r", type=int, default=16)
    g.add_argument("--peft-lora-alpha", type=int, default=32)
    g.add_argument("--peft-lora-dropout", type=float, default=0.05)
    g.add_argument("--gradient-checkpointing", action="store_true")

    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", required=True,
                   help="processed Dolly split (training) or a benchmark directory (evaluation)")
    g.add_argument("--dev-num", type=int, default=-1, help="validation examples used for checkpoint selection")
    g.add_argument("--max-length", type=int, default=512)
    g.add_argument("--max-prompt-length", type=int, default=256)
    g.add_argument("--num-workers", type=int, default=1)

    g = parser.add_argument_group("training")
    g.add_argument("--save", required=True, help="output directory")
    g.add_argument("--seed", type=int, default=10)
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=8, help="per-GPU micro batch")
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)
    g.add_argument("--eval-batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--lr-min", type=float, default=1e-7, help="final learning rate of the cosine schedule")
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--clip-grad", type=float, default=1.0)
    g.add_argument("--log-interval", type=int, default=4)

    g = parser.add_argument_group("distillation")
    g.add_argument("--loss", default="sft", choices=LOSSES)
    g.add_argument("--drkl-gamma", type=float, default=1.0, help="gamma: DRKL weight of the non-target term")
    g.add_argument("--drkl-gamma-start", type=float,
                   help="DRKL-linear: grow gamma linearly from this value to --drkl-gamma over training")
    g.add_argument("--skew-alpha", type=float, default=0.1, help="skew of SFKL, SRKL and SRKL + DRKL")
    g.add_argument("--kd-ratio", type=float, default=1.0,
                   help="loss = kd_ratio * distillation + (1 - kd_ratio) * SFT cross-entropy")
    g.add_argument("--log-rkl-terms", action="store_true",
                   help="log q_m, the target / non-target KL terms of RKL and d NRKL / d z_m")

    g = parser.add_argument_group("on-policy (DistiLLM adaptive student-generated outputs)")
    g.add_argument("--on-policy", action="store_true")
    g.add_argument("--init-threshold", type=float, default=0.5, help="initial probability of using student samples")
    g.add_argument("--loss-eps", type=float, default=0.1,
                   help="raise the threshold by 0.1 when validation loss grows by more than this")
    g.add_argument("--capacity", type=int, default=1000, help="replay buffer size")
    g.add_argument("--student-gen-max-new-tokens", type=int, default=None)

    g = parser.add_argument_group("generation")
    g.add_argument("--eval-max-new-tokens", type=int, default=None,
                   help="cap on generated tokens (default: max-length - max-prompt-length)")

    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.loss != "sft" and args.teacher_model_path is None:
        parser.error(f"--loss {args.loss} needs --teacher-model-path")
    return args
