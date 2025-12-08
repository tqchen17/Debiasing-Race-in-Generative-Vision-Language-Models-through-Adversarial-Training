# Running on Brown Oscar - Setup Guide

## Summary

Your project downloads ~20GB of COCO-2017 data and ~100MB of ResNet50 weights. This guide ensures safe execution on Oscar without crashes or quota issues.

## Data Requirements

1. **COCO-2017 Dataset**: ~20GB
   - Train images: ~18GB (118K images)
   - Validation images: ~1GB (5K images)
   - Annotations: ~500MB

2. **ResNet50 Weights**: ~100MB
   - Auto-downloaded by Keras on first run
   - Cached in `~/.keras/models/`

3. **Total Space Needed**: ~25GB (with headroom)

## Pre-Flight Checklist

### 1. Check Your Oscar Quotas

```bash
# SSH to Oscar
ssh youruser@sshcampus.ccv.brown.edu

# Check all quotas
checkquota
```

**From your quota check**:
- Home (`~/`): 100GB total - **Too small for data (only 18GB free)**
- Scratch (`~/scratch`): 512GB soft quota - **Perfect! Use this!**

**IMPORTANT**: Do NOT use your home directory (`~/`) for the dataset! Use `~/scratch/` instead.

## Installation Steps

### Step 1: Initial Setup on Oscar

```bash
# SSH to Oscar
ssh youruser@sshcampus.ccv.brown.edu

# Create project directory in scratch space
mkdir -p ~/scratch/debiasing-project
cd ~/scratch/debiasing-project

# Clone your repo or copy files
# Option A: If using git
git clone <your-repo-url> .

# Option B: Copy from home if already there
cp -r ~/Debiasing-Race-in-Generative-Vision-Language-Models-through-Adversarial-Training/* .
```

### Step 2: Set Up Python Environment

```bash
# Load Python module
module load python/3.11.0

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 3: Download Data (Choose ONE option)

#### Option A: Download on Oscar (Recommended)

```bash
# Make sure scripts are executable
chmod +x oscar_download.sh

# Submit download job (runs in background, won't crash if you disconnect)
sbatch oscar_download.sh

# Monitor progress
tail -f download_*.log

# Check if job is running
squeue -u $USER
```

This will take 2-3 hours. The job runs in the background, so you can disconnect.

#### Option B: Download Locally, Then Transfer

If you have fast internet locally:

```bash
# On your LOCAL machine
python preprocess.py  # Downloads to data/fiftyone/

# Transfer to Oscar (this will take a while)
scp -r data/ youruser@sshcampus.ccv.brown.edu:~/scratch/debiasing-project/
scp MASTER_TRAIN.csv MASTER_VAL.csv youruser@sshcampus.ccv.brown.edu:~/scratch/debiasing-project/
```

### Step 4: Verify Data

```bash
# Check that data downloaded correctly
ls -lh ~/scratch/debiasing-project/data/fiftyone/coco-2017/train/data/ | head
ls -lh ~/scratch/debiasing-project/data/fiftyone/coco-2017/validation/data/ | head

# Check CSV files exist
ls -lh ~/scratch/debiasing-project/MASTER_TRAIN.csv
ls -lh ~/scratch/debiasing-project/MASTER_VAL.csv

# Verify vocab.pkl exists
ls -lh ~/scratch/debiasing-project/vocab.pkl

# Check total size
du -sh ~/scratch/debiasing-project
```

## Running Training

### Submit Training Job

```bash
# Make sure you're in the project directory
cd ~/scratch/debiasing-project

# Make script executable
chmod +x oscar_train.sh

# Submit training job
sbatch oscar_train.sh

# Monitor progress
tail -f train_*.log

# Check GPU usage (if job is running)
squeue -u $USER
```

### Monitor Training

```bash
# View logs in real-time
tail -f train_*.log

# Check TensorBoard logs (optional, from your local machine)
ssh -L 6006:localhost:6006 youruser@sshcampus.ccv.brown.edu
# Then on Oscar:
cd ~/scratch/debiasing-project
source .venv/bin/activate
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

## Troubleshooting

### "Disk quota exceeded"

**Problem**: Your home directory is full.

**Solution**: Make sure you're using `~/scratch/`, not `~/`

```bash
# Check where you are
pwd
# Should show: /oscar/scratch/youruser/debiasing-project
# NOT: /users/youruser/... or /oscar/home/youruser/...

# If in wrong location, move everything
mv ~/debiasing-project ~/scratch/
```

### Download Keeps Failing

**Problem**: Network timeout during download.

**Solution**: The updated `preprocess.py` now skips re-downloading if data exists. FiftyOne also has built-in resume capability.

If download fails partway, just re-run:
```bash
sbatch oscar_download.sh
```

It will resume from where it left off.

### ResNet50 Weights Not Downloading

**Problem**: Keras can't download pretrained weights.

**Solution**: Pre-download on login node:

```bash
# On Oscar login node (NOT in a job)
module load python/3.11.0
cd ~/scratch/debiasing-project
source .venv/bin/activate

# This will download ResNet50 weights to ~/.keras/models/
python -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet')"
```

### Out of Memory During Training

**Problem**: GPU runs out of memory.

**Solution**: Reduce batch size in `oscar_train.sh`:

```bash
# Edit the file
nano oscar_train.sh

# Change line:
#   --batch_size 32 \
# To:
#   --batch_size 16 \
```

### Job Times Out After 24 Hours

**Problem**: Training isn't finishing in 24 hours.

**Solution**: The code saves checkpoints every 5 epochs. To resume:

1. Check which checkpoint exists:
   ```bash
   ls -lh ~/scratch/debiasing-project/checkpoints/
   ```

2. Training automatically resumes from latest checkpoint when you re-run:
   ```bash
   sbatch oscar_train.sh
   ```

## File Structure on Oscar

After setup, your directory should look like:

```
~/scratch/debiasing-project/
├── data/
│   └── fiftyone/
│       └── coco-2017/
│           ├── train/
│           │   └── data/           # ~18GB of images
│           └── validation/
│               └── data/           # ~1GB of images
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── checkpoints/                    # Model checkpoints
├── logs/                          # TensorBoard logs
├── MASTER_TRAIN.csv
├── MASTER_VAL.csv
├── vocab.pkl
├── oscar_download.sh
├── oscar_train.sh
└── requirements.txt
```

## Best Practices

1. **Always use `~/scratch/`** for large datasets (you have 512GB!)
2. **Use sbatch for long jobs** - never run training on login nodes
3. **Request appropriate time** - 24 hours for full training
4. **Monitor disk usage** - run `du -sh ~/scratch/debiasing-project` periodically
5. **Clean up old checkpoints** - keep only best model to save space
6. **Use TensorBoard** - monitor training curves

## Important Note About Scratch

**WARNING**: Files in `~/scratch/` are automatically deleted after 30 days of inactivity!

To preserve your work after training completes:
```bash
# Copy final model to home (has backups)
mkdir -p ~/model-backups
cp -r ~/scratch/debiasing-project/checkpoints ~/model-backups/

# Or download to your local machine
# (Run this on YOUR machine, not Oscar)
scp -r youruser@sshcampus.ccv.brown.edu:~/scratch/debiasing-project/checkpoints ./
```

## Quick Commands Reference

```bash
# Submit download job
sbatch oscar_download.sh

# Submit training job
sbatch oscar_train.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check disk usage
du -sh ~/scratch/debiasing-project

# Check quotas
checkquota

# View logs
tail -f train_*.log
tail -f download_*.log

# Copy important results to home before auto-deletion (30 days)
cp -r ~/scratch/debiasing-project/checkpoints ~/model-backups/
```

## Space Management

Based on your quota:
- **Home** (`~/`): 100GB - Keep code and final models here
- **Scratch** (`~/scratch`): 512GB - Use for datasets and training

Recommended workflow:
```bash
# Code lives in home
~/Debiasing-Race-in-Generative-Vision-Language-Models-through-Adversarial-Training/

# Data and training in scratch
~/scratch/debiasing-project/

# After training, copy best model back to home
cp -r ~/scratch/debiasing-project/checkpoints/best_model ~/model-backups/
```

## Getting Help

- Oscar documentation: https://docs.ccv.brown.edu/oscar/
- CS1470 staff: Use EdStem or office hours
- CCV support: support@ccv.brown.edu
