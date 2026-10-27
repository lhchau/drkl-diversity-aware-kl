#!/usr/bin/env bash
# Download the benchmarks and the checkpoints of one model family, then tokenize Dolly.
#
#   bash scripts/prepare_data.sh qwen3|llama3|opt|gpt2
#
# Llama-3.2 is gated on the Hugging Face Hub: accept its license and run `hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/.."

FAMILY=${1:?usage: bash scripts/prepare_data.sh qwen3|llama3|opt|gpt2}

download() { hf download "$1" --local-dir "$2" "${@:3}"; }

# instruction-following benchmarks released with MiniLLM (Gu et al., 2024)
for name in dolly self-inst sinst uinst; do
  download "MiniLLM/$name" "data/$name" --repo-type dataset
done
download MiniLLM/Vicuna data/vicuna --repo-type dataset

case $FAMILY in
  qwen3)
    download Qwen/Qwen3-4B checkpoints/Qwen3-4B
    download Qwen/Qwen3-32B checkpoints/Qwen3-32B
    TOKENIZER=checkpoints/Qwen3-4B ;;
  llama3)
    download meta-llama/Llama-3.2-1B checkpoints/Llama-3.2-1B
    download meta-llama/Llama-3.2-3B checkpoints/Llama-3.2-3B
    TOKENIZER=checkpoints/Llama-3.2-1B ;;
  opt)
    download MiniLLM/init-OPT-1.3B checkpoints/opt-1.3b     # OPT-1.3B initialized on Dolly by MiniLLM
    download MiniLLM/SFT-OPT-6.7B checkpoints/opt-6.7b      # OPT-6.7B fine-tuned on Dolly
    TOKENIZER=checkpoints/opt-1.3b ;;
  gpt2)
    download gpt2 checkpoints/gpt2-base
    download gpt2-medium checkpoints/gpt2-medium
    download MiniLLM/teacher-gpt2-1.5B checkpoints/gpt2-xl  # GPT-2 XL fine-tuned on Dolly
    TOKENIZER=checkpoints/gpt2-base ;;
  *) echo "unknown family: $FAMILY" >&2; exit 1 ;;
esac

python3 tools/process_data_dolly.py --model-path "$TOKENIZER" --model-type "$FAMILY"
if [ "$FAMILY" = qwen3 ]; then
  # Qwen3 is prompted through its chat template, also at evaluation time
  python3 tools/process_data_dolly.py --model-path "$TOKENIZER" --model-type qwen3 \
    --eval-sets data/dolly data/self-inst data/vicuna data/sinst/11_ data/uinst/11_
fi
