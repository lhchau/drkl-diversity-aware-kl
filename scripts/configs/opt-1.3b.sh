# OPT 6.7B -> OPT 1.3B, full fine-tuning.
MODEL_TYPE=opt
MODEL=checkpoints/opt-1.3b
TEACHER=checkpoints/opt-6.7b
LR=5e-5
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=32
MAX_LENGTH=512
DEV_NUM=1000
DS_CONFIG=configs/deepspeed/ds_config_zero1_fp16.json
GAMMA=${GAMMA:-1.0}
TRAIN_OPTS="--gradient-checkpointing"
