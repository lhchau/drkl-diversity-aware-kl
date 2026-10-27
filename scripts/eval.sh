#!/usr/bin/env bash
# Evaluate a checkpoint on the five benchmarks with five sampling seeds and print
# mean +- std of ROUGE-L and the diversity metrics per benchmark.
#
#   bash scripts/eval.sh <config> <checkpoint> [extra evaluate.py args]
#
#   bash scripts/eval.sh qwen3-4b results/qwen3-4b/drkl-gamma1.5-seed10/best_rougeL
#   bash scripts/eval.sh qwen3-32b results/qwen3-32b/sft-seed10/best_rougeL   # the teacher
#
# LoRA checkpoints (directories with adapter_config.json) are loaded on top of
# the config's MODEL. Results go to <checkpoint>/eval/<benchmark>/seed<seed>.
# Override BENCHMARKS="dolly vicuna" or SEEDS="10" to run a subset.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then sed -n '2,12p' "$0"; exit 1; fi
CONFIG=$1 CKPT=${2%/}
shift 2
source "scripts/configs/$CONFIG.sh"

BENCHMARKS=${BENCHMARKS:-"dolly self-inst vicuna sinst/11_ uinst/11_"}
SEEDS=${SEEDS:-"10 20 30 40 50"}
NGPU=${NGPU:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
if (( NGPU < 1 )); then echo "no CUDA GPU visible" >&2; exit 1; fi
if [ -f "$CKPT/adapter_config.json" ]; then
  MODEL_ARGS="--model-path $MODEL --peft lora --peft-path $CKPT"
else
  MODEL_ARGS="--model-path $CKPT"
fi

export TOKENIZERS_PARALLELISM=false
for bench in $BENCHMARKS; do
  # the LoRA runs capped generation on the two large sets (Super-NI 1.7k, UnNI 24k prompts)
  CAP=""
  if [[ -n "${EVAL_CAP:-}" && $bench =~ ^(sinst|uinst) ]]; then CAP="--eval-max-new-tokens $EVAL_CAP"; fi
  for seed in $SEEDS; do
    torchrun --standalone --nproc_per_node "$NGPU" evaluate.py $MODEL_ARGS --model-type "$MODEL_TYPE" \
      --data-dir "data/$bench" --max-length "$MAX_LENGTH" --max-prompt-length 256 \
      --eval-batch-size "$EVAL_BATCH_SIZE" --seed "$seed" --save "$CKPT/eval/$bench/seed$seed" \
      --deepspeed_config "$DS_CONFIG" $CAP "$@"
  done
done

python3 tools/summarize_eval.py "$CKPT/eval"
