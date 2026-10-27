# Qwen3-32B -> Qwen3-4B, LoRA student.
MODEL_TYPE=qwen3
MODEL=checkpoints/Qwen3-4B
TEACHER=checkpoints/Qwen3-32B
TEACHER_PEFT=results/qwen3-32b/sft-seed10/best_rougeL
LR=1.5e-4
EPOCHS=10
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=16
MAX_LENGTH=1024
DEV_NUM=200
DS_CONFIG=configs/deepspeed/ds_config_zero2_bf16.json
GAMMA=${GAMMA:-1.5}
# the Qwen3 runs average the distillation and SFT losses
TRAIN_OPTS="--peft lora --gradient-checkpointing --eval-max-new-tokens 256 --kd-ratio 0.5"
EVAL_CAP=256
