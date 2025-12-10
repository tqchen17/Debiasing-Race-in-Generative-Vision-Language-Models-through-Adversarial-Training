#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

# Load required modules
module load python/3.11.0
module load cuda/11.8.0
module load cudnn/8.9.0

# Set working directory (use scratch directory, not home)
export WORK_DIR=~/scratch/debiasing-project
cd $WORK_DIR

# Activate virtual environment
source .venv/bin/activate

# Set TensorFlow to use GPU efficiently
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Verify GPU is available
python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"

echo "Starting training..."
echo "Working directory: $WORK_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run training (adjust epochs and batch_size as needed)
python src/training/baseline_train.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --checkpoint_dir $WORK_DIR/checkpoints \
    --log_dir $WORK_DIR/logs

echo "Training complete!"
