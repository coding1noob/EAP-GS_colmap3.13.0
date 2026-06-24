#!/bin/bash
#SBATCH -J step4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /data2/buaajianghengchen/storageCode/EAP-GS/logs/step4-%j.out
#SBATCH -e /data2/buaajianghengchen/storageCode/EAP-GS/logs/step4-%j.err

set -euo pipefail

mkdir -p /data2/buaajianghengchen/storageCode/EAP-GS/logs
cd /data2/buaajianghengchen/storageCode/EAP-GS

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

source ~/.bashrc
source /home/buaajianghengchen/anaconda3/etc/profile.d/conda.sh
conda activate /data2/buaajianghengchen/conda_envs/EAP-GS

python train.py \
  -s /data2/buaajianghengchen/datasets/EAP-GS/buaa1 \
  --model_path /data2/buaajianghengchen/storageCode/EAP-GS/output \
  --seed 3 \
  --pc_name points3D_600views_aug \
  --eval \
  --visualize_points \
  --points_shrink 1024