#!/bin/bash

################################################################################
# VQ-VAE Training Script
# 
# This script provides convenient training modes and configurations for the VQ-VAE
# model on AMEX time-series data.
#
# Usage:
#   ./train.sh [MODE] [OPTIONS]
#
# Modes:
#   debug       - Quick training on small dataset for debugging
#   quick       - Quick training with reduced epochs
#   full        - Full training with default settings
#   custom      - Custom training with user-provided arguments
#
# Examples:
#   ./train.sh debug
#   ./train.sh full --use-wandb
#   ./train.sh custom --num-epochs 100 --batch-size 64 --use-wandb
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/train_vq_vae.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/checkpoints_${TIMESTAMP}"

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python script exists
check_script_exists() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Training script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Check Python and dependencies
check_dependencies() {
    print_info "Checking Python and dependencies..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_info "Using Python $python_version"
    
    # Check for PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_warning "PyTorch not detected. Please install it before training."
        exit 1
    fi
    
    # Check for wandb (optional)
    if python -c "import wandb" 2>/dev/null; then
        print_info "Wandb is available"
    else
        print_warning "Wandb not installed (optional, but needed for --use-wandb)"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
${BLUE}VQ-VAE Training Script${NC}

${GREEN}Usage:${NC}
  $0 [MODE] [OPTIONS]

${GREEN}Modes:${NC}
  debug       Quick debugging with small dataset (100 samples, 2 epochs)
  quick       Quick training (500 samples, 5 epochs)
  full        Full training with default settings
  custom      Custom training with arbitrary arguments
  
${GREEN}Model Hyperparameters:${NC}
  --input-dim DIM              Input feature dimension (default: 512)
  --num-embeddings N           Number of codebook embeddings (default: 512)
  --embedding-dim DIM          Dimension of codebook embeddings (default: 64)
  --commitment-cost COST       Commitment cost for VQ loss (default: 0.25)
  --encoder-num-layers N       Number of encoder transformer layers (default: 2)
  --encoder-num-heads N        Number of encoder attention heads (default: 8)
  --decoder-num-layers N       Number of decoder transformer layers (default: 2)
  --decoder-num-heads N        Number of decoder attention heads (default: 8)
  --dropout RATE               Dropout rate for all layers (default: 0.1)

${GREEN}Training Hyperparameters:${NC}
  --batch-size N               Batch size for training (default: 32)
  --val-batch-size N           Batch size for validation (default: 64)
  --num-epochs N               Number of training epochs (default: 20)
  --learning-rate LR           Learning rate (default: 0.001)
  --weight-decay WD            Weight decay for optimizer (default: 0.0001)
  --num-workers N              Number of data loading workers (default: 4)
  --reconstruction-loss-weight W    Weight for reconstruction loss (default: 1.0)
  --commitment-loss-weight W        Weight for commitment loss (default: 0.25)

${GREEN}Logging and Checkpointing:${NC}
  --output-dir PATH            Directory to save checkpoints (default: auto)
  --checkpoint-freq N          Save checkpoint every N epochs (default: 1)
  --log-freq N                 Log metrics every N batches (default: 10)
  --save-best                  Save best model based on validation loss

${GREEN}Weights & Biases Logging:${NC}
  --use-wandb                  Enable Weights & Biases logging
  --wandb-project NAME         Wandb project name (default: vq-vae-amex)
  --wandb-tags TAG1 TAG2       Add tags to Wandb run
  --wandb-notes NOTES          Notes for the Wandb run

${GREEN}Other Options:${NC}
  --device cuda|cpu            Device to use (default: cuda)
  --resume-from PATH           Resume from checkpoint
  --seed SEED                  Random seed (default: 42)
  --presets                    Show hyperparameter tuning presets
  --help                       Show this help message

${GREEN}Examples:${NC}
  # Debug mode with custom model
  $0 debug --num-embeddings 256 --embedding-dim 128 --dropout 0.2
  
  # Full training with tuned hyperparameters
  $0 full --learning-rate 5e-4 --batch-size 64 --num-epochs 50 --save-best
  
  # Custom with specific model and training config
  $0 custom \\
    --encoder-num-layers 4 --encoder-num-heads 16 \\
    --decoder-num-layers 4 --decoder-num-heads 16 \\
    --batch-size 48 --learning-rate 2e-4 \\
    --num-epochs 100 --use-wandb --wandb-tags large-model
  
  # Advanced tuning with Wandb
  $0 full \\
    --num-embeddings 1024 --embedding-dim 128 \\
    --dropout 0.15 --learning-rate 5e-4 \\
    --reconstruction-loss-weight 1.5 \\
    --use-wandb --wandb-tags tuning-v1

EOF
}

# Run debug mode
run_debug() {
    print_info "Running in DEBUG mode"
    print_info "Base Settings: 100 samples, 2 epochs, batch_size=16"
    
    if [ $# -gt 0 ]; then
        print_info "Additional arguments: $@"
    fi
    
    python "$PYTHON_SCRIPT" \
        --output-dir "$DEFAULT_OUTPUT_DIR" \
        --debug \
        --debug-size 100 \
        --batch-size 16 \
        --num-epochs 2 \
        --num-workers 2 \
        --log-freq 5 \
        "$@"
}

# Run quick mode
run_quick() {
    print_info "Running in QUICK mode"
    print_info "Base Settings: 500 samples, 5 epochs, batch_size=32"
    
    if [ $# -gt 0 ]; then
        print_info "Additional arguments: $@"
    fi
    
    python "$PYTHON_SCRIPT" \
        --output-dir "$DEFAULT_OUTPUT_DIR" \
        --debug \
        --debug-size 500 \
        --batch-size 32 \
        --num-epochs 5 \
        --num-workers 4 \
        --log-freq 10 \
        "$@"
}

# Run full training
run_full() {
    print_info "Running in FULL mode"
    print_info "Base Settings: Full dataset, 20 epochs, batch_size=32"
    
    if [ $# -gt 0 ]; then
        print_info "Additional arguments: $@"
    fi
    
    python "$PYTHON_SCRIPT" \
        --output-dir "$DEFAULT_OUTPUT_DIR" \
        --batch-size 32 \
        --num-epochs 20 \
        --num-workers 4 \
        --log-freq 10 \
        --save-best \
        "$@"
}

# Run custom training
run_custom() {
    print_info "Running in CUSTOM mode"
    
    if [ $# -gt 0 ]; then
        print_info "Arguments: $@"
    fi
    
    python "$PYTHON_SCRIPT" \
        --output-dir "$DEFAULT_OUTPUT_DIR" \
        "$@"
}

# Show hyperparameter tuning presets
show_presets() {
    cat << EOF
${BLUE}VQ-VAE Hyperparameter Tuning Presets${NC}

${GREEN}Small Model:${NC}
./train.sh full \\
  --num-embeddings 256 --embedding-dim 32 \\
  --encoder-num-layers 1 --encoder-num-heads 4 \\
  --decoder-num-layers 1 --decoder-num-heads 4 \\
  --dropout 0.05 --batch-size 64

${GREEN}Large Model:${NC}
./train.sh full \\
  --num-embeddings 1024 --embedding-dim 128 \\
  --encoder-num-layers 4 --encoder-num-heads 16 \\
  --decoder-num-layers 4 --decoder-num-heads 16 \\
  --dropout 0.2 --batch-size 32

${GREEN}Conservative Learning:${NC}
./train.sh full \\
  --learning-rate 5e-4 --weight-decay 5e-4 \\
  --commitment-loss-weight 0.5 \\
  --batch-size 32 --num-epochs 50

${GREEN}Aggressive Learning:${NC}
./train.sh full \\
  --learning-rate 2e-3 --weight-decay 1e-5 \\
  --commitment-loss-weight 0.1 \\
  --batch-size 64 --num-epochs 20

${GREEN}Balanced:${NC}
./train.sh full \\
  --num-embeddings 512 --embedding-dim 64 \\
  --encoder-num-layers 2 --encoder-num-heads 8 \\
  --decoder-num-layers 2 --decoder-num-heads 8 \\
  --learning-rate 1e-3 --batch-size 32 \\
  --dropout 0.1 --num-epochs 30 --save-best

${GREEN}Memory-Efficient:${NC}
./train.sh full \\
  --num-embeddings 256 --embedding-dim 48 \\
  --encoder-num-layers 1 --encoder-num-heads 4 \\
  --decoder-num-layers 1 --decoder-num-heads 4 \\
  --batch-size 64 --num-workers 2

${GREEN}Reconstruction-Focused:${NC}
./train.sh full \\
  --reconstruction-loss-weight 2.0 \\
  --commitment-loss-weight 0.1 \\
  --learning-rate 5e-4 --batch-size 32

${GREEN}Quantization-Focused:${NC}
./train.sh full \\
  --reconstruction-loss-weight 0.5 \\
  --commitment-loss-weight 0.5 \\
  --learning-rate 1e-3 --batch-size 32

${BLUE}Usage:${NC}
Copy and paste any preset, then add your own options or use with --use-wandb

EOF
}

# Main script logic
main() {
    # Show help if requested
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        show_usage
        exit 0
    fi
    
    # Show presets if requested  
    if [ "$1" == "--presets" ]; then
        show_presets
        exit 0
    fi
    
    # Show help if no arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    # Check prerequisites
    check_script_exists
    check_dependencies
    
    # Get mode (default is 'full')
    MODE="${1:-full}"
    shift || true  # Shift remaining arguments
    
    print_info "=========================================="
    print_info "VQ-VAE Training Script"
    print_info "=========================================="
    print_info "Mode: $MODE"
    print_info "Output Directory: $DEFAULT_OUTPUT_DIR"
    print_info "Timestamp: $TIMESTAMP"
    print_info ""
    
    # Create output directory
    mkdir -p "$DEFAULT_OUTPUT_DIR"
    
    # Run appropriate mode
    case "$MODE" in
        debug)
            run_debug "$@"
            ;;
        quick)
            run_quick "$@"
            ;;
        full)
            run_full "$@"
            ;;
        custom)
            run_custom "$@"
            ;;
        *)
            print_error "Unknown mode: $MODE"
            echo ""
            show_usage
            exit 1
            ;;
    esac
    
    # Check exit status
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
        print_info "Results saved to: $DEFAULT_OUTPUT_DIR"
        exit 0
    else
        print_error "Training failed with exit code $?"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
