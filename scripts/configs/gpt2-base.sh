# GPT-2 XL (1.5B) -> GPT-2 Base (120M), full fine-tuning (Appendix F.2).
MODEL_TYPE=gpt2
MODEL=checkpoints/gpt2-base
TEACHER=checkpoints/gpt2-xl
LR=5e-4
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=32
MAX_LENGTH=512
DEV_NUM=1000
DS_CONFIG=configs/deepspeed/ds_config_zero1_fp16.json
GAMMA=${GAMMA:-0.5}
TRAIN_OPTS=""
