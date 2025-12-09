#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=eval_%j.log
#SBATCH --error=eval_%j.err

# Set working directory
export WORK_DIR=~/scratch/debiasing-project
cd $WORK_DIR

# Activate virtual environment
source .venv/bin/activate

# Set TensorFlow to use GPU efficiently
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting evaluation on multiple splits..."
echo "Working directory: $WORK_DIR"
echo "This will replicate Table 1 from 'Balanced Datasets Are Not Enough'"

# Run evaluation on validation set with multiple balanced splits
# This evaluates: original, α=2, and α=1 (perfectly balanced)
python src/training/evaluate_baseline_all_splits.py \
    --checkpoint $WORK_DIR/checkpoints/best_model.weights.h5 \
    --batch_size 16 \
    --output_dir $WORK_DIR/results \
    --splits original,alpha_2,alpha_1 \
    --use_val

echo "Evaluation complete!"
echo "Results saved to: $WORK_DIR/results"
echo ""
echo "Check the output table to see if 'Balanced Datasets Are Not Enough'!"
echo "Key question: Does Δ (bias amplification) remain high even with α=1 (balanced)?"
