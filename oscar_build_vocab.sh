#!/bin/bash
#SBATCH --job-name=build_vocab
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=build_vocab_%j.log
#SBATCH --error=build_vocab_%j.err

# Load required modules
module load python/3.11.0

# Set data directory to scratch space
export DATA_DIR=~/scratch/debiasing-project

# Navigate to project directory
cd $DATA_DIR

# Activate virtual environment
source .venv/bin/activate

echo "Building vocabulary from MASTER_TRAIN.csv..."

# Run vocabulary builder
python src/utils/vocab.py \
    --train_csv MASTER_TRAIN.csv \
    --min_freq 5 \
    --save_path vocab.pkl

echo "Vocabulary building complete!"
echo "vocab.pkl saved to: $DATA_DIR/vocab.pkl"
