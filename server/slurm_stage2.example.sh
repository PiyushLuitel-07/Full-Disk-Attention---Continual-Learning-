#!/usr/bin/env bash
# Submit only after Stage 1 completed in the same configured run directory.
#SBATCH --job-name=hmi-ewc-s2
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
  --stage 2 \
  --device cuda

