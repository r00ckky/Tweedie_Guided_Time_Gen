#!/usr/bin/env bash
set -euo pipefail
# Latent Diffusion (DiT) Training Script on AMEX Time-Series Data
# Trains a Diffusion Transformer atop a frozen VQ-VAE codebook space

# ============================================================================
# Configuration Variables
# ============================================================================

# Dataset Paths and Configuration
DATA_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/data/Amex_data"
TRAIN_DATA="${DATA_DIR}/train_data.csv"
TRAIN_LABELS="${DATA_DIR}/train_labels.csv"
DB_PATH="${DATA_DIR}/amex_data.db"
MAX_SEQ_LEN=14
BATCH_SIZE=256
VAL_BATCH_SIZE=256
NUM_WORKERS=8
CLASS_IMBALANCE=true

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

VQVAE_CHECKPOINT="/home/chaitanya-kohli/Amex/TimeVQDM/checkpoints/knn_0.5_l12h8_VQ512/best_model.pt"

DIT_HIDDEN_DIM=256
DIT_COND_DIM=256
DIT_NUM_LAYERS=6
DIT_NUM_HEADS=8
DIT_FF_MULTIPLIER=4
DIT_DROPOUT=0.1
DIT_NUM_CLASSES=2
CFG_DROPOUT_PROB=0.10

# Diffusion Schedule
TIMESTEPS=1000
SCHEDULE="cosine"

# Training Hyperparameters
NUM_EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=1.0
WARMUP_EPOCHS=5
LOG_FREQ=10
CHECKPOINT_FREQ=10
SAVE_BEST=true
SEED=42

# Output and Logging
OUTPUT_DIR="/home/chaitanya-kohli/Amex/TimeVQDM/dit_output/DiT_l${DIT_NUM_LAYERS}h${DIT_NUM_HEADS}_T${TIMESTEPS}"
mkdir -p "${OUTPUT_DIR}"

WANDB_PROJECT="amex-latent-diffusion"
WANDB_RUN_NAME="DiT_l${DIT_NUM_LAYERS}h${DIT_NUM_HEADS}_T${TIMESTEPS}_$(date +%d%m_%H%M%S)"
WANDB_SAVE_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_SAVE_DIR}"

DEVICE="cuda"
DEBUG=false
RESUME_FROM=""

# ============================================================================
# Build Command
# ============================================================================

cd /home/chaitanya-kohli/Amex/TimeVQDM
export PYTHONPATH="/home/chaitanya-kohli/Amex/TimeVQDM:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python train_latent_diffusion.py \
    --output-dir "${OUTPUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --train-data "${TRAIN_DATA}" \
    --train-labels "${TRAIN_LABELS}" \
    --db-path "${DB_PATH}" \
    --max-seq-len ${MAX_SEQ_LEN} \
    $([ "${CLASS_IMBALANCE}" = true ] && echo "--class_imbalance") \
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
    --vqvae-checkpoint "${VQVAE_CHECKPOINT}" \
    --dit-hidden-dim ${DIT_HIDDEN_DIM} \
    --dit-cond-dim ${DIT_COND_DIM} \
    --dit-num-layers ${DIT_NUM_LAYERS} \
    --dit-num-heads ${DIT_NUM_HEADS} \
    --dit-ff-multiplier ${DIT_FF_MULTIPLIER} \
    --dit-dropout ${DIT_DROPOUT} \
    --dit-num-classes ${DIT_NUM_CLASSES} \
    --cfg-dropout-prob ${CFG_DROPOUT_PROB} \
    --timesteps ${TIMESTEPS} \
    --schedule ${SCHEDULE} \
    --batch-size ${BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --max-grad-norm ${MAX_GRAD_NORM} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --num-workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --log-freq ${LOG_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} \
    $([ "${SAVE_BEST}" = true ] && echo "--save-best") \
    --device ${DEVICE} \
    $([ "${DEBUG}" = true ] && echo "--debug") \
    $([ -n "${RESUME_FROM}" ] && echo "--resume-from ${RESUME_FROM}") \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-save-dir "${WANDB_SAVE_DIR}"

echo "DiT Training completed! Checkpoints saved to ${OUTPUT_DIR}"