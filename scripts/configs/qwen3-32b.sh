# Qwen3-32B teacher: LoRA SFT on Dolly (chat template, thinking disabled).
MODEL_TYPE=qwen3
MODEL=checkpoints/Qwen3-32B
LR=5e-5
EPOCHS=3
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=16
MAX_LENGTH=1024
DEV_NUM=200
DS_CONFIG=configs/deepspeed/ds_config_zero2_bf16.json
TRAIN_OPTS="--peft lora --gradient-checkpointing --eval-max-new-tokens 256"
EVAL_CAP=256
