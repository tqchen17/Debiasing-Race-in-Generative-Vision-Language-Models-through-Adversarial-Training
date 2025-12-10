#!/bin/bash
#SBATCH --job-name=adv_dual_train
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=train_adv_%j.log
#SBATCH --error=train_adv_%j.err

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

echo "======================================================================"
echo "DUAL ADVERSARIAL DEBIASING TRAINING"
echo "======================================================================"
echo "Working directory: $WORK_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Model Configuration:"
echo "  - Gradient reversal at: encoder features + decoder hidden states"
echo "  - Lambda encoder (λ_v): 0.5 (default, adjustable)"
echo "  - Lambda decoder (λ_d): 0.5 (default, adjustable)"
echo ""
echo "Loss Function:"
echo "  L = L_caption - λ_v * L_adv_encoder - λ_d * L_adv_decoder"
echo ""
echo "Expected Behavior:"
echo "  - Caption loss should decrease (good captions)"
echo "  - Adversary losses should increase (race becomes unpredictable)"
echo "  - This indicates successful debiasing!"
echo "======================================================================"
echo ""

# Run adversarial training
# NOTE: Hyperparameters are set in src/utils/config.py
#       Modify config.py to change:
#         - LAMBDA_V, LAMBDA_D (adversarial weights)
#         - LR_ENCODER, LR_DECODER, LR_ADVERSARY (learning rates)
#         - BATCH_SIZE, NUM_EPOCHS
python src/training/adv_train_dual.py

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check TensorBoard: tensorboard --logdir=$WORK_DIR/logs/adversarial_dual"
echo "  2. Evaluate bias: python src/training/evaluate_baseline_all_splits.py \\"
echo "                     --checkpoint $WORK_DIR/checkpoints/adversarial_dual/best_model.weights.h5"
echo "  3. Compare with baseline results to measure bias reduction"
echo ""
echo "Expected Results (from 'Balanced Datasets Are Not Enough'):"
echo "  - Bias amplification (Δ) should DECREASE"
echo "  - Caption quality (F1) should remain similar (small drop acceptable)"
echo "  - Trade-off: ~53-67% bias reduction with ~1-2 F1 point loss"
echo "======================================================================"
