#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# NCSN (Score Matching) Training Script on AMEX Time-Series Data
# ============================================================================

# 1. Dataset Paths
DATA_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/data/Amex_data"
TRAIN_DATA="${DATA_DIR}/train_data.csv"
TRAIN_LABELS="${DATA_DIR}/train_labels.csv"
DB_PATH="${DATA_DIR}/amex_data.db"

# 2. Preprocessing Artifacts
TRANSFORMER_PATH="/home/chaitanya-kohli/Amex/TimeVQDM/quantile_transformer.pkl"
FILL_DICT_PATH="/home/chaitanya-kohli/Amex/TimeVQDM/fill_dict.pkl"

# 3. Target Pre-trained VQ-VAE Checkpoint
VQ_VAE_CHECKPOINT="/home/chaitanya-kohli/Amex/TimeVQDM/checkpoints/knn_0.5_l12h8_VQ512/best_model.pt"

# 4. NCSN Specific Configuration
DENOISER_HIDDEN_DIM=256
LEARNING_RATE=1e-3  # Standard for diffusion/score models 
NUM_EPOCHS=100
BATCH_SIZE=64
VAL_BATCH_SIZE=64
NCSN_NUM_BLOCKS=6

# 5. Original VQ-VAE Architecture Config 
MAX_SEQ_LEN=14
INPUT_DIM=147
PATCH_SIZE=2
PATCH_STRIDE=2
PATCH_EMBED_DIM=256
NUM_EMBEDDINGS=512
EMBEDDING_DIM=256
HIDDEN_DIM=256
NUM_LAYERS=12
NUM_HEADS=8
FF_MULTIPLIER=4
DROPOUT=0.1
USE_CLASS_TOKEN=true
CLASS_PROJ_DIM=1
CLASS_FUNC='knn'

NUM_WORKERS=8
WEIGHT_DECAY=1e-4
LOG_FREQ=10
CHECKPOINT_FREQ=2
SEED=42
DEVICE="cuda"
CLASS_IMBALANCE=true

CHECKPOINT_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/ncsn_checkpoints/NCSN_denoiser_${DENOISER_HIDDEN_DIM}_l_${NCSN_NUM_BLOCKS}"
mkdir -p "${CHECKPOINT_DIR}"

WANDB_PROJECT="ncsn-amex"
WANDB_RUN_NAME="NCSN_denoiser_${DENOISER_HIDDEN_DIM}_l_${NCSN_NUM_BLOCKS}_$(date +%d%m_%H%M%S)"
WANDB_SAVE_DIR="${CHECKPOINT_DIR}/wandb"
mkdir -p "${WANDB_SAVE_DIR}"

NCSN_WEIGHTS=""

cd /home/chaitanya-kohli/Amex/TimeVQDM
export PYTHONPATH="/home/chaitanya-kohli/Amex/TimeVQDM:${PYTHONPATH:-}"

echo "Starting NCSN Training..."
echo "Using VQ-VAE Checkpoint: ${VQ_VAE_CHECKPOINT}"

CUDA_VISIBLE_DEVICES=0 python train_ncsn.py \
    --output-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --train-data "${TRAIN_DATA}" \
    --train-labels "${TRAIN_LABELS}" \
    --db-path "${DB_PATH}" \
    --transformer_path "${TRANSFORMER_PATH}" \
    --fill-dict-path "${FILL_DICT_PATH}" \
    --vq-vae-checkpoint "${VQ_VAE_CHECKPOINT}" \
    --denoiser-hidden-dim ${DENOISER_HIDDEN_DIM} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --input-dim ${INPUT_DIM} \
    --patch-size ${PATCH_SIZE} \
    --patch-stride ${PATCH_STRIDE} \
    --patch-embed-dim ${PATCH_EMBED_DIM} \
    --num-embeddings ${NUM_EMBEDDINGS} \
    --embedding-dim ${EMBEDDING_DIM} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --num-heads ${NUM_HEADS} \
    --ff-multiplier ${FF_MULTIPLIER} \
    --dropout ${DROPOUT} \
    $([ "${USE_CLASS_TOKEN}" = true ] && echo "--use-class-token") \
    --class-proj-dim ${CLASS_PROJ_DIM} \
    --class_func ${CLASS_FUNC} \
    --batch-size ${BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --num-workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --log-freq ${LOG_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} \
    --device ${DEVICE} \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-save-dir "${WANDB_SAVE_DIR}" \
    --use-wandb \
    $([ "${CLASS_IMBALANCE}" = true ] && echo "--class_imbalance") \
    $([ -f "${NCSN_WEIGHTS}" ] && echo "--ncsn_weights ${NCSN_WEIGHTS}") \
    --ncsn_num_blocks ${NCSN_NUM_BLOCKS} \

echo "NCSN Training completed! Checkpoints saved to ${CHECKPOINT_DIR}"