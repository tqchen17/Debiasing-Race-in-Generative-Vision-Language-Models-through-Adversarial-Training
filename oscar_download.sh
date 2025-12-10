#!/bin/bash
#SBATCH --job-name=coco_download
#SBATCH --time=20:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=download_%j.log
#SBATCH --error=download_%j.err

# Load required modules
module load python/3.11.0
module load cuda/11.8.0

# Set data directory to scratch space (has 512GB quota)
export DATA_DIR=~/scratch/debiasing-project

# Navigate to project directory (assumes you already cloned repo here)
cd $DATA_DIR

# Activate virtual environment
source .venv/bin/activate

# Set FiftyOne cache directory
export FIFTYONE_DATASET_DIR=$DATA_DIR/data/fiftyone

echo "Starting COCO download to $FIFTYONE_DATASET_DIR"
echo "This may take 2-3 hours for the full dataset"

# Run preprocessing script
python preprocess.py

echo "Download complete!"
echo "Data saved to: $DATA_DIR"
