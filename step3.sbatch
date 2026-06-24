#!/bin/bash
#SBATCH -J step3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /data2/buaajianghengchen/storageCode/EAP-GS/logs/step3-%j.out
#SBATCH -e /data2/buaajianghengchen/storageCode/EAP-GS/logs/step3-%j.err

set -euo pipefail

mkdir -p /data2/buaajianghengchen/storageCode/EAP-GS/logs
cd /data2/buaajianghengchen/storageCode/EAP-GS

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

source ~/.bashrc
source /home/buaajianghengchen/anaconda3/etc/profile.d/conda.sh
conda activate /data2/buaajianghengchen/conda_envs/colmap37

python run_colmap.py \
  --base_path /data2/buaajianghengchen/datasets/EAP-GS \
  --scene buaa1 \
  --views 600 \
  --augment
