#!/bin/bash
#SBATCH --job-name=eval_adv
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=eval_adv_%j.log
#SBATCH --error=eval_adv_%j.err

# Set working directory
export WORK_DIR=~/scratch/debiasing-project
cd $WORK_DIR

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
which python
python --version

# Set TensorFlow to use GPU efficiently
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

echo "======================================================================"
echo "ADVERSARIAL MODEL EVALUATION"
echo "======================================================================"
echo "Working directory: $WORK_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Evaluating adversarial debiasing model on multiple data splits:"
echo "  - original split (unbalanced, ~2.3:1 light:dark)"
echo "  - α = 2.0 split (up to 2:1 light:dark)"
echo "  - α = 1.0 split (perfectly balanced, 1:1 light:dark)"
echo ""
echo "Comparing against baseline results to measure bias reduction."
echo "======================================================================"
echo ""

# Run evaluation on all splits
# Using the latest TensorFlow checkpoint (check checkpoints/adversarial_dual/ for latest ckpt-XX number)
echo "About to run Python script..."
echo "Command: python src/training/evaluate_adversarial_all_splits.py --checkpoint $WORK_DIR/checkpoints/adversarial_dual/ckpt-35 --batch_size 16 --splits original,alpha_2,alpha_1 --use_val"

python -u src/training/evaluate_adversarial_all_splits.py --checkpoint $WORK_DIR/checkpoints/adversarial_dual/ckpt-35 --batch_size 16 --splits original,alpha_2,alpha_1 --use_val

PYTHON_EXIT=$?
echo "Python script exited with code: $PYTHON_EXIT"

if [ $PYTHON_EXIT -ne 0 ]; then
    echo "ERROR: Python script failed!"
    exit $PYTHON_EXIT
fi

echo ""
echo "======================================================================"
echo "EVALUATION COMPLETE!"
echo "======================================================================"
echo ""
echo "Checkpoint evaluated: checkpoints/adversarial_dual/ckpt-35"
echo ""
echo "Results saved to:"
echo "  - $WORK_DIR/results/adversarial_all_splits_*.json"
echo "  - $WORK_DIR/results/adversarial_table_all_splits_*.txt"
echo ""
echo "Next steps:"
echo "  1. Compare adversarial results with baseline results"
echo "  2. Check if bias amplification (Δ) decreased"
echo "  3. Verify caption quality (F1) remained acceptable"
echo ""
echo "Expected improvements:"
echo "  - Δ should be LOWER than baseline (53-67% reduction)"
echo "  - F1 may drop by ~1-2 points (acceptable trade-off)"
echo "  - This confirms adversarial debiasing is working!"
echo "======================================================================"
