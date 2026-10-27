#!/usr/bin/env bash
# Train one model on Dolly.
#
#   bash scripts/train.sh <config> <loss> [seed] [extra finetune.py args]
#
#   bash scripts/train.sh qwen3-32b sft                    # LoRA SFT of the Qwen3-32B teacher
#   bash scripts/train.sh qwen3-4b drkl                    # Qwen3-32B -> Qwen3-4B with DRKL
#   GAMMA=0.5 bash scripts/train.sh qwen3-4b drkl 20       # another gamma and seed
#   bash scripts/train.sh llama3-1b drkl 10 --on-policy    # on-policy distillation
#
# <config> is a file in scripts/configs, <loss> one of
# sft fkl rkl symkl dkd js sfkl srkl akl ab drkl srkl_drkl. DRKL's gamma comes from
# the config unless GAMMA is set. All visible GPUs are used and the global batch is
# kept at 32 by gradient accumulation. The run is saved to
# results/<config>/<loss>[-gamma<gamma>]-seed<seed>[-linear][-onpolicy].
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then sed -n '2,16p' "$0"; exit 1; fi
CONFIG=$1 LOSS=$2 SEED=10
if [[ ${3:-} =~ ^[0-9]+$ ]]; then SEED=$3; shift 3; else shift 2; fi
source "scripts/configs/$CONFIG.sh"
GAMMA=${GAMMA:-1.0}

NGPU=${NGPU:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
if (( NGPU < 1 )); then echo "no CUDA GPU visible" >&2; exit 1; fi
GLOBAL_BATCH=32
GRAD_ACC=$(( GLOBAL_BATCH / (BATCH_SIZE * NGPU) ))
if (( GRAD_ACC < 1 || GRAD_ACC * BATCH_SIZE * NGPU != GLOBAL_BATCH )); then
  echo "global batch $GLOBAL_BATCH is not a multiple of BATCH_SIZE=$BATCH_SIZE x $NGPU GPUs" >&2
  exit 1
fi

RUN=$LOSS-seed$SEED
if [[ $LOSS == *drkl ]]; then RUN=$LOSS-gamma$GAMMA-seed$SEED; fi
if [[ " $* " == *" --drkl-gamma-start "* ]]; then RUN=$RUN-linear; fi
if [[ " $* " == *" --on-policy "* ]]; then RUN=$RUN-onpolicy; fi
SAVE=results/$CONFIG/$RUN

export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc_per_node "$NGPU" finetune.py \
  --model-type "$MODEL_TYPE" --model-path "$MODEL" \
  ${TEACHER:+--teacher-model-path "$TEACHER"} ${TEACHER_PEFT:+--teacher-peft-path "$TEACHER_PEFT"} \
  --data-dir "processed_data/dolly/full/$MODEL_TYPE" --dev-num "$DEV_NUM" \
  --max-length "$MAX_LENGTH" --max-prompt-length 256 --num-workers 4 \
  --loss "$LOSS" --drkl-gamma "$GAMMA" --seed "$SEED" \
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACC" \
  --eval-batch-size "$EVAL_BATCH_SIZE" --save "$SAVE" \
  --deepspeed --deepspeed_config "$DS_CONFIG" \
  $TRAIN_OPTS "$@"
