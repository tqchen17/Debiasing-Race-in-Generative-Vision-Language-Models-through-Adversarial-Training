"""
Configuration file for hyperparameters and settings.
"""

import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================
# Data Paths
# ======================
TRAIN_CSV = os.path.join(PROJECT_ROOT, 'MASTER_TRAIN.csv')
VAL_CSV = os.path.join(PROJECT_ROOT, 'MASTER_VAL.csv')
VOCAB_PATH = os.path.join(PROJECT_ROOT, 'vocab.pkl')

# ======================
# Data Parameters
# ======================
IMAGE_SIZE = 224  # ResNet50 input size
IMAGE_CHANNELS = 3
MAX_CAPTION_LENGTH = 20  # Maximum number of words in caption
VOCAB_MIN_FREQ = 5  # Minimum word frequency for vocabulary

# ======================
# Model Architecture
# ======================
# Encoder
ENCODER_TYPE = 'resnet50'  # tf.keras.applications.ResNet50
ENCODER_DIM = 2048  # ResNet50 output dimension
FINETUNE_ENCODER = False  # Start with frozen encoder

# Decoder
DECODER_TYPE = 'lstm'  # LSTM with attention
EMBED_DIM = 300  # Word embedding dimension
DECODER_DIM = 512  # LSTM hidden dimension
ATTENTION_DIM = 512  # Attention layer dimension
DROPOUT = 0.5

# Adversary Networks
ADVERSARY_V_DIMS = [2048, 512, 512, 512, 2]  # Encoder adversary dimensions
ADVERSARY_D_DIMS = [512, 256, 256, 2]  # Decoder adversary dimensions

# ======================
# Training Hyperparameters
# ======================
# Learning rates
LR_ENCODER = 1e-4  # Learning rate for encoder (if finetuning)
LR_DECODER = 1e-3  # Learning rate for decoder
LR_ADVERSARY = 1e-3  # Learning rate for adversaries

# Adversarial weights
LAMBDA_V = 0.5  # Encoder adversary weight (tune: 0.3-1.0)
LAMBDA_D = 0.5  # Decoder adversary weight (tune: 0.3-1.0)

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 80
GRAD_CLIP = 5.0  # Gradient clipping threshold

# ======================
# Data Augmentation
# ======================
RANDOM_FLIP = True  # Random horizontal flip
RANDOM_CROP = True  # Random crop after resize

# ImageNet normalization (for ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ======================
# Evaluation
# ======================
EVAL_BATCH_SIZE = 64
EVAL_METRICS = ['bleu', 'rouge', 'cider']

# ======================
# Checkpoints and Logging
# ======================
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Adversarial model directories
DUAL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'adversarial_dual')
DUAL_LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'adversarial_dual')

# Temporal adversary setting
USE_TEMPORAL_ADVERSARY = False  # Use temporal averaging for decoder adversary

# Save frequency
SAVE_EVERY_N_EPOCHS = 5
LOG_EVERY_N_STEPS = 100

# ======================
# Race Labels
# ======================
RACE_LABELS = {
    'Light': 0,
    'Dark': 1
}
NUM_RACE_CLASSES = 2

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, DUAL_CHECKPOINT_DIR, DUAL_LOG_DIR]:
    os.makedirs(directory, exist_ok=True)


def print_config():
    """Print configuration settings."""
    print("=" * 60)
    print("Configuration Settings")
    print("=" * 60)
    print(f"\nData Paths:")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Val CSV: {VAL_CSV}")
    print(f"  Vocab: {VOCAB_PATH}")

    print(f"\nModel Architecture:")
    print(f"  Encoder: {ENCODER_TYPE} (dim={ENCODER_DIM})")
    print(f"  Decoder: {DECODER_TYPE} (dim={DECODER_DIM})")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Attention dim: {ATTENTION_DIM}")

    print(f"\nTraining:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LR (encoder): {LR_ENCODER}")
    print(f"  LR (decoder): {LR_DECODER}")
    print(f"  LR (adversary): {LR_ADVERSARY}")
    print(f"  Lambda_v: {LAMBDA_V}")
    print(f"  Lambda_d: {LAMBDA_D}")

    print(f"\nData:")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Max caption length: {MAX_CAPTION_LENGTH}")
    print(f"  Vocab min freq: {VOCAB_MIN_FREQ}")

    print("=" * 60)


if __name__ == '__main__':
    print_config()
