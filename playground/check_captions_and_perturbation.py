"""
Quick diagnostic to understand the bias metrics behavior.
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path - handle both relative and absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add src directory
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.dataset import create_datasets
from models.baseline_model import create_baseline_model
from utils.vocab import Vocabulary
import utils.config as config


def main():
    print("="*80)
    print("CAPTION & PERTURBATION DIAGNOSTIC")
    print("="*80)

    # Load vocab
    print("\nLoading vocabulary...")
    vocab = Vocabulary.load(config.VOCAB_PATH)
    print(f"Vocab size: {len(vocab)}")

    # Load dataset
    print("\nLoading dataset...")
    _, val_dataset, _ = create_datasets(batch_size=16, balanced=False)

    # Load model
    print("\nLoading model...")
    model = create_baseline_model(vocab_size=len(vocab), finetune_encoder=False)

    # Build
    dummy_batch = next(iter(val_dataset))
    _ = model(dummy_batch['image'], dummy_batch['caption'], training=False)

    # Load weights
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.weights.h5")
    print(f"Loading weights from {checkpoint_path}...")
    model.load_weights(checkpoint_path)

    # Get one batch
    print("\n" + "="*80)
    print("SAMPLE CAPTIONS (First 10)")
    print("="*80)

    batch = next(iter(val_dataset))
    images = batch['image'][:10]
    gt_captions = batch['caption'][:10]
    race_labels = batch['race_label'][:10]

    for i in range(10):
        # Ground truth
        gt_text = vocab.decode(gt_captions[i].numpy())

        # Generated
        img_batch = tf.expand_dims(images[i], 0)
        generated_ids = model.generate_caption(
            img_batch,
            start_token=vocab.start_idx,
            end_token=vocab.end_idx,
            max_length=20
        )
        gen_text = vocab.decode(generated_ids)

        race = "Light" if race_labels[i].numpy() == 0 else "Dark"

        print(f"\n[{i+1}] Race: {race}")
        print(f"  GT:  {gt_text}")
        print(f"  GEN: {gen_text}")

    # Test perturbation
    print("\n" + "="*80)
    print("PERTURBATION TEST")
    print("="*80)

    # Get a sample caption
    sample_gt = vocab.decode(gt_captions[0].numpy())
    print(f"\nOriginal caption: '{sample_gt}'")

    # Test different perturbation rates
    test_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    for rate in test_rates:
        words = sample_gt.split()
        perturbed_words = []

        for word in words:
            if np.random.random() < rate:
                random_idx = np.random.randint(0, len(vocab))
                random_word = vocab.idx2word.get(random_idx, '<UNK>')
                perturbed_words.append(random_word)
            else:
                perturbed_words.append(word)

        perturbed = ' '.join(perturbed_words)
        print(f"  Rate {rate:.1f}: '{perturbed}'")

    # Show what F1=0.35 means
    print("\n" + "="*80)
    print("WHAT DOES F1=0.3469 MEAN?")
    print("="*80)
    print("\nYour model achieves F1=0.3469, which is quite low.")
    print("This means it's getting roughly 1/3 of words correct.")
    print("\nTo perturb ground truth to F1=0.35, we need to replace ~65% of words.")
    print("At that level of corruption, patterns might still leak through TF-IDF features.")

    # Check model quality stats
    print("\n" + "="*80)
    print("CHECKING MODEL QUALITY")
    print("="*80)

    from evaluation.caption_metrics import evaluate_captions

    generated_all = []
    ground_truth_all = []

    for i in range(10):
        gt_text = vocab.decode(gt_captions[i].numpy())

        img_batch = tf.expand_dims(images[i], 0)
        generated_ids = model.generate_caption(
            img_batch,
            start_token=vocab.start_idx,
            end_token=vocab.end_idx,
            max_length=20
        )
        gen_text = vocab.decode(generated_ids)

        generated_all.append(gen_text)
        ground_truth_all.append(gt_text)

    references = [[gt] for gt in ground_truth_all]
    metrics = evaluate_captions(generated_all, references)

    print(f"\nCaption Quality (10 samples):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print("\n1. If generated captions are mostly gibberish:")
    print("   → Model is undertrained, F1 is too low")
    print("   → Perturbation at F1=0.35 is too aggressive")
    print("   → λD(F1) ≈ λD because heavy perturbation preserves TF-IDF patterns")
    print("\n2. If generated captions are coherent:")
    print("   → Model is working, but there might be a methodology issue")
    print("\n3. Expected behavior:")
    print("   → λM should be HIGHER than λD(F1) if model amplifies bias")
    print("   → Your α=1 results DO show this (Δ=0.0945)")
    print("   → But original split shows Δ=0.0000, which is unusual")
    print("\nRecommendation:")
    print("  - If captions look reasonable, proceed to adversarial debiasing")
    print("  - If captions are gibberish, consider training longer first")
    print("="*80)


if __name__ == '__main__':
    main()
