#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=preprocess_%j.log
#SBATCH --error=preprocess_%j.err

# Load required modules
module load python/3.11.0

# Set data directory to scratch space
export DATA_DIR=~/scratch/debiasing-project

# Navigate to project directory
cd $DATA_DIR

# Activate virtual environment
source .venv/bin/activate

echo "Starting preprocessing..."
echo "Working directory: $DATA_DIR"
echo "This will skip download and generate MASTER_TRAIN.csv and MASTER_VAL.csv"

# Run preprocessing script (will skip download if data exists)
python preprocess.py

echo "Preprocessing complete!"
echo "CSV files saved to: $DATA_DIR"
