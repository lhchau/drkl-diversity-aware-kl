# Llama-3.2-3B teacher: LoRA SFT on Dolly.
MODEL_TYPE=llama3
MODEL=checkpoints/Llama-3.2-3B
LR=1e-4
EPOCHS=10
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=8
MAX_LENGTH=1024
DEV_NUM=200
DS_CONFIG=configs/deepspeed/ds_config_zero2_bf16.json
TRAIN_OPTS="--peft lora --gradient-checkpointing --eval-max-new-tokens 256"
EVAL_CAP=256
