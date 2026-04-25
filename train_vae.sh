#!/usr/bin/env bash
set -euo pipefail
# VAE Training Script on AMEX Time-Series Data
# Trains a Variational Autoencoder with optional classification

# ============================================================================
# Configuration Variables
# ============================================================================

# Dataset Paths and Configuration
DATA_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/data/Amex_data"
TRAIN_DATA="${DATA_DIR}/train_data.csv"
TRAIN_LABELS="${DATA_DIR}/train_labels.csv"
DB_PATH="${DATA_DIR}/amex_data.db"
MAX_SEQ_LEN=13
BATCH_SIZE=64
NUM_WORKERS=8
NUM_EPOCHS=20
CLASS_IMBALANCE=true
CLASS_FUNC='knn'
VAL_BATCH_SIZE=256

# Model Architecture Configuration - Streamlined
# Patch Embedding
INPUT_DIM=147
PATCH_SIZE=2
PATCH_STRIDE=2
PATCH_EMBED_DIM=256

# Latent Space
LATENT_DIM=256

# Transformer (Unified for Encoder and Decoder)
HIDDEN_DIM=256
NUM_LAYERS=12
NUM_HEADS=8
FF_MULTIPLIER=4
DROPOUT=0.1

# Classification
USE_CLASS_TOKEN=true
CLASS_PROJ_DIM=1

# Training Hyperparameters
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
VAL_BATCH_SIZE=64
LOG_FREQ=10
CHECKPOINT_FREQ=1
SAVE_BEST=true
SEED=42

# Loss Weights
RECONSTRUCTION_LOSS_WEIGHT=1.0
KL_LOSS_WEIGHT=1.0
CLASSIFICATION_LOSS_WEIGHT=0.5

# Output and Logging
CHECKPOINT_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/checkpoints/FixLN_${CLASS_FUNC}_${CLASSIFICATION_LOSS_WEIGHT}_l${NUM_LAYERS}h${NUM_HEADS}_lat${LATENT_DIM}"
mkdir -p "${CHECKPOINT_DIR}"

# Weights & Biases Configuration
WANDB_PROJECT="vae-amex"
WANDB_ENTITY=""
WANDB_RUN_NAME="FixLN_${CLASS_FUNC}_${CLASSIFICATION_LOSS_WEIGHT}_l${NUM_LAYERS}h${NUM_HEADS}_lat${LATENT_DIM}_$(date +%d%m_%H%M%S)"
WANDB_SAVE_DIR="${CHECKPOINT_DIR}/wandb"
WANDB_NOTES=""
WANDB_TAGS=()
mkdir -p "${WANDB_SAVE_DIR}"

# Device Configuration
DEVICE="cuda"
USE_FP16=false

# Debugging Configuration
DEBUG=false
DEBUG_SIZE=100
RESUME_FROM=""

# ============================================================================
# Build Command
# ============================================================================

cd /home/chaitanya-kohli/Amex/TimeVQDM
export PYTHONPATH="/home/chaitanya-kohli/Amex/TimeVQDM:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python train_vae.py \
    --output-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --train-data "${TRAIN_DATA}" \
    --train-labels "${TRAIN_LABELS}" \
    --db-path "${DB_PATH}" \
    --max-seq-len ${MAX_SEQ_LEN} \
    --input-dim ${INPUT_DIM} \
    --patch-size ${PATCH_SIZE} \
    --patch-stride ${PATCH_STRIDE} \
    --patch-embed-dim ${PATCH_EMBED_DIM} \
    --latent-dim ${LATENT_DIM} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --num-heads ${NUM_HEADS} \
    --ff-multiplier ${FF_MULTIPLIER} \
    --dropout ${DROPOUT} \
    $([ "${USE_CLASS_TOKEN}" = true ] && echo "--use-class-token") \
    --class-proj-dim ${CLASS_PROJ_DIM} \
    --batch-size ${BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --num-workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --log-freq ${LOG_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} \
    --reconstruction-loss-weight ${RECONSTRUCTION_LOSS_WEIGHT} \
    --kl-loss-weight ${KL_LOSS_WEIGHT} \
    --classification-loss-weight ${CLASSIFICATION_LOSS_WEIGHT} \
    $([ "${SAVE_BEST}" = true ] && echo "--save-best") \
    --device ${DEVICE} \
    $([ "${USE_FP16}" = true ] && echo "--fp16") \
    $([ "${DEBUG}" = true ] && echo "--debug") \
    $([ "${DEBUG}" = true ] && echo "--debug-size ${DEBUG_SIZE}") \
    $([ -n "${RESUME_FROM}" ] && echo "--resume-from ${RESUME_FROM}") \
    --wandb-project "${WANDB_PROJECT}" \
    $([ -n "${WANDB_ENTITY}" ] && echo "--wandb-entity '${WANDB_ENTITY}'") \
    $([ -n "${WANDB_NOTES}" ] && echo "--wandb-notes '${WANDB_NOTES}'") \
    $([ ${#WANDB_TAGS[@]} -gt 0 ] && echo "--wandb-tags ${WANDB_TAGS[@]}") \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-save-dir "${WANDB_SAVE_DIR}" \
    --use-wandb \
    $([ "${CLASS_IMBALANCE}" = true ] && echo "--class_imbalance") \
    --class_func ${CLASS_FUNC} \
    --val-batch-size ${VAL_BATCH_SIZE} \

echo "VAE Training completed! Checkpoints saved to ${CHECKPOINT_DIR}"
