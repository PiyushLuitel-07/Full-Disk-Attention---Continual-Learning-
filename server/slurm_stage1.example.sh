#!/usr/bin/env bash
# Example only. Use this file only if NOVA actually provides sbatch, and replace
# the account/partition/time values with the local administrator's instructions.
#SBATCH --job-name=hmi-ewc-s1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /CHANGE/THIS/TO/fulldiskattention_continual
source .venv/bin/activate
python -m modeling.train_continual \
  --config configs/server.json \
  --stage 1 \
  --device cuda

