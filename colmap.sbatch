#!/bin/bash
#SBATCH -J colmap-garden
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /data2/buaajianghengchen/storageCode/EAP-GS/logs/%x-%j.out
#SBATCH -e /data2/buaajianghengchen/storageCode/EAP-GS/logs/%x-%j.err

set -euo pipefail

mkdir -p /data2/buaajianghengchen/storageCode/EAP-GS/logs

SCENE_ROOT=/data2/buaajianghengchen/datasets/EAP-GS/garden
IMAGE_PATH=${SCENE_ROOT}/images
DB_PATH=${SCENE_ROOT}/database.db
SPARSE_ROOT=${SCENE_ROOT}/sparse

echo "==== Job info ===="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

source ~/.bashrc
source /home/buaajianghengchen/anaconda3/etc/profile.d/conda.sh
conda activate /data2/buaajianghengchen/conda_envs/EAP-GS

echo "==== Paths ===="
echo "SCENE_ROOT=${SCENE_ROOT}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "DB_PATH=${DB_PATH}"
echo "SPARSE_ROOT=${SPARSE_ROOT}"

echo "==== Cleaning previous COLMAP outputs ===="
rm -f "${DB_PATH}"
rm -rf "${SPARSE_ROOT}"
mkdir -p "${SPARSE_ROOT}"

echo "==== 1. Running COLMAP feature_extractor ===="
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.max_image_size 1600 \
    --SiftExtraction.estimate_affine_shape 0 \
    --SiftExtraction.domain_size_pooling 0

echo "==== 2. Running COLMAP exhaustive_matcher ===="
colmap exhaustive_matcher \
    --database_path "${DB_PATH}" \
    --FeatureMatching.use_gpu 1

echo "==== 3. Running COLMAP mapper ===="
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --output_path "${SPARSE_ROOT}"

echo "==== 4. Checking mapper outputs ===="
find "${SPARSE_ROOT}" -maxdepth 2 -type f | sort

echo "==== Done ===="
