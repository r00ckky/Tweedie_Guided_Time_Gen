#!/usr/bin/env bash
set -euo pipefail
# VQ-VAE Training Script on AMEX Time-Series Data
# Trains a Vector Quantized Variational Autoencoder with optional classification

# ============================================================================
# Configuration Variables
# ============================================================================

# Dataset Paths and Configuration
DATA_DIR="/scratch/s25090/Amex_data"
TRAIN_DATA="${DATA_DIR}/train_data.csv"
TRAIN_LABELS="${DATA_DIR}/train_labels.csv"
DB_PATH="${DATA_DIR}/amex.db"
MAX_SEQ_LEN=13
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=20

# Model Architecture Configuration
INPUT_DIM=512
NUM_EMBEDDINGS=512
EMBEDDING_DIM=64
COMMITMENT_COST=0.25
ENCODER_LAYERS=2
ENCODER_HEADS=8
DECODER_LAYERS=2
DECODER_HEADS=8
DROPOUT=0.1
ENCODER_CLASS_TOKEN=false
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
COMMITMENT_LOSS_WEIGHT=0.25
CLASSIFICATION_LOSS_WEIGHT=0.5

# Classification Configuration
CLASSIFICATION_ENABLED=true

# Output and Logging
CHECKPOINT_DIR="/scratch/s25090/vq_vae/checkpoints_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CHECKPOINT_DIR}"

# Weights & Biases Configuration
WANDB_PROJECT="vq-vae-amex"
WANDB_ENTITY=""
WANDB_RUN_NAME="vq_vae_run_$(date +%Y%m%d_%H%M%S)"
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

CUDA_VISIBLE_DEVICES=0 python train_vq_vae.py \
    --output-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --train-data "${TRAIN_DATA}" \
    --train-labels "${TRAIN_LABELS}" \
    --db-path "${DB_PATH}" \
    --max-seq-len ${MAX_SEQ_LEN} \
    --input-dim ${INPUT_DIM} \
    --num-embeddings ${NUM_EMBEDDINGS} \
    --embedding-dim ${EMBEDDING_DIM} \
    --commitment-cost ${COMMITMENT_COST} \
    --encoder-num-layers ${ENCODER_LAYERS} \
    --encoder-num-heads ${ENCODER_HEADS} \
    --decoder-num-layers ${DECODER_LAYERS} \
    --decoder-num-heads ${DECODER_HEADS} \
    --dropout ${DROPOUT} \
    $([ "${ENCODER_CLASS_TOKEN}" = true ] && echo "--encoder_class_token") \
    --encoder-class-proj-dim ${CLASS_PROJ_DIM} \
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
    --commitment-loss-weight ${COMMITMENT_LOSS_WEIGHT} \
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
    --use-wandb

echo "Training completed! Checkpoints saved to ${CHECKPOINT_DIR}"
