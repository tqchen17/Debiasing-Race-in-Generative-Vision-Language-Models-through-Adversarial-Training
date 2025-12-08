"""
Comprehensive evaluation script for baseline model.
Measures both caption quality AND bias metrics.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_datasets
from models.baseline_model import create_baseline_model
from evaluation.caption_metrics import evaluate_captions, print_metrics
from evaluation.bias_metrics import evaluate_model_bias, print_bias_summary
from utils.vocab import Vocabulary
import utils.config as config


def evaluate_baseline(model, dataset, vocab, num_batches=None, adversary_epochs=20):
    """
    Comprehensive evaluation: Caption quality + Bias metrics.

    Args:
        model: Trained baseline model
        dataset: Evaluation dataset
        vocab: Vocabulary
        num_batches: Number of batches to evaluate (None = all)
        adversary_epochs: Epochs to train adversary probe

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BASELINE EVALUATION")
    print("=" * 80)

    results = {}

    # Collect all predictions
    all_generated = []
    all_references = []
    all_race_labels = []
    all_encoder_features = []
    all_decoder_hiddens = []

    print("\nGenerating captions...")

    batch_count = 0
    for batch in dataset:
        images = batch['image']
        captions = batch['caption']
        race_labels = batch['race_label']

        # Forward pass to get features
        predictions, encoder_features, decoder_hiddens = model(
            images, captions, training=False
        )

        # Store features for bias evaluation
        all_encoder_features.append(encoder_features.numpy())

        # Average decoder hiddens across timesteps
        decoder_avg = tf.reduce_mean(decoder_hiddens, axis=1)
        all_decoder_hiddens.append(decoder_avg.numpy())

        # Generate captions
        for i in range(len(images)):
            single_image = tf.expand_dims(images[i], 0)

            # Generate
            generated_ids = model.generate_caption(
                single_image,
                start_token=vocab.start_idx,
                end_token=vocab.end_idx,
                max_length=20
            )

            # Decode
            generated_text = vocab.decode(generated_ids)
            reference_text = vocab.decode(captions[i].numpy())

            all_generated.append(generated_text)
            all_references.append([reference_text])

        all_race_labels.extend(race_labels.numpy().tolist())

        batch_count += 1
        if num_batches and batch_count >= num_batches:
            break

        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...")

    print(f"\nTotal samples evaluated: {len(all_generated)}")

    # Concatenate features
    encoder_features = np.concatenate(all_encoder_features, axis=0)
    decoder_features = np.concatenate(all_decoder_hiddens, axis=0)
    race_labels = np.array(all_race_labels)

    # ==========================================
    # 1. CAPTION QUALITY METRICS
    # ==========================================
    print("\n" + "=" * 80)
    print("1. CAPTION QUALITY EVALUATION")
    print("=" * 80)

    caption_results = evaluate_captions(all_generated, all_references)
    print_metrics(caption_results, "Caption Quality Metrics")

    results['caption_quality'] = caption_results

    # ==========================================
    # 2. BIAS METRICS
    # ==========================================
    print("\n" + "=" * 80)
    print("2. BIAS EVALUATION")
    print("=" * 80)

    from evaluation.bias_metrics import BiasMetrics

    bias_metrics = BiasMetrics()

    # 2a. Adversary probing on encoder
    print("\n2a. Encoder Adversary Probing")
    print("-" * 80)
    encoder_adv_results = bias_metrics.train_adversary_probe(
        encoder_features,
        race_labels,
        input_dim=encoder_features.shape[1],
        epochs=adversary_epochs
    )
    results['encoder_adversary'] = encoder_adv_results

    # 2b. Adversary probing on decoder
    print("\n2b. Decoder Adversary Probing")
    print("-" * 80)
    decoder_adv_results = bias_metrics.train_adversary_probe(
        decoder_features,
        race_labels,
        input_dim=decoder_features.shape[1],
        epochs=adversary_epochs
    )
    results['decoder_adversary'] = decoder_adv_results

    # 2c. Racial word frequency
    print("\n2c. Racial Descriptor Analysis")
    print("-" * 80)
    race_word_results = bias_metrics.compute_race_word_frequency(
        all_generated,
        all_race_labels
    )
    results['race_words'] = race_word_results

    print(f"  Light captions - racial word freq: {race_word_results['light_racial_word_freq']:.4f}")
    print(f"  Dark captions - racial word freq: {race_word_results['dark_racial_word_freq']:.4f}")
    print(f"  Disparity: {race_word_results['racial_word_disparity']:.4f}")

    # 2d. Per-race caption quality
    print("\n2d. Per-Race Caption Quality")
    print("-" * 80)
    per_race_results = bias_metrics.compute_per_race_quality(
        all_generated,
        all_references,
        all_race_labels
    )
    results['per_race_quality'] = per_race_results

    if 'light' in per_race_results:
        print(f"  Light - BLEU-4: {per_race_results['light']['BLEU-4']:.4f} "
              f"({per_race_results['light_count']} samples)")
    if 'dark' in per_race_results:
        print(f"  Dark - BLEU-4: {per_race_results['dark']['BLEU-4']:.4f} "
              f"({per_race_results['dark_count']} samples)")
    if 'bleu4_disparity' in per_race_results:
        print(f"  Quality disparity: {per_race_results['bleu4_disparity']:.4f}")

    # 2e. Dataset statistics
    light_count = np.sum(race_labels == 0)
    dark_count = np.sum(race_labels == 1)
    total = len(race_labels)

    results['dataset_stats'] = {
        'light_ratio': float(light_count / total),
        'dark_ratio': float(dark_count / total),
        'light_count': int(light_count),
        'dark_count': int(dark_count),
        'total': int(total)
    }

    print("\n2e. Dataset Statistics")
    print("-" * 80)
    print(f"  Light samples: {light_count} ({light_count/total*100:.1f}%)")
    print(f"  Dark samples: {dark_count} ({dark_count/total*100:.1f}%)")

    # 2f. Bias amplification
    bias_amp = bias_metrics.compute_bias_amplification(
        encoder_adv_results['adversary_accuracy'],
        {'light': light_count / total}
    )
    results['bias_amplification'] = float(bias_amp)

    print("\n2f. Bias Amplification")
    print("-" * 80)
    print(f"  Amplification index: {bias_amp:.4f}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print_bias_summary(results, "BASELINE MODEL - BIAS EVALUATION SUMMARY")

    return results


def save_results(results, output_dir):
    """
    Save evaluation results to file.

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'baseline_eval_{timestamp}.json')

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results_clean = convert_types(results)

    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save a summary
    summary_file = os.path.join(output_dir, f'baseline_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("BASELINE MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("CAPTION QUALITY:\n")
        f.write("-" * 80 + "\n")
        for key, value in results['caption_quality'].items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nBIAS METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Encoder Adversary Accuracy: {results['encoder_adversary']['adversary_accuracy']:.4f}\n")
        f.write(f"  Decoder Adversary Accuracy: {results['decoder_adversary']['adversary_accuracy']:.4f}\n")

        if 'per_race_quality' in results:
            pr = results['per_race_quality']
            if 'light' in pr and 'dark' in pr:
                f.write(f"  Light BLEU-4: {pr['light']['BLEU-4']:.4f}\n")
                f.write(f"  Dark BLEU-4: {pr['dark']['BLEU-4']:.4f}\n")
                if 'quality_gap' in pr:
                    f.write(f"  Quality Gap: {pr['quality_gap']:+.4f}\n")

        f.write(f"\n  Bias Amplification: {results['bias_amplification']:.4f}\n")

    print(f"Summary saved to: {summary_file}")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate baseline model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint or weights')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Number of batches to evaluate (None = all)')
    parser.add_argument('--adversary_epochs', type=int, default=20,
                       help='Epochs to train adversary probes')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--use_val', action='store_true',
                       help='Evaluate on validation set (default: train)')

    args = parser.parse_args()

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Vocabulary.load(config.VOCAB_PATH)

    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = create_datasets(
        batch_size=args.batch_size,
        balanced=False  # Use natural distribution for evaluation
    )

    dataset = val_dataset if args.use_val else train_dataset
    dataset_name = "Validation" if args.use_val else "Training"

    print(f"\nEvaluating on {dataset_name} set")

    # Create model
    print("\nCreating model...")
    model = create_baseline_model(vocab_size=len(vocab), finetune_encoder=False)

    # Load weights
    print(f"Loading weights from {args.checkpoint}...")
    model.load_weights(args.checkpoint).expect_partial()
    print("Weights loaded successfully!")

    # Evaluate
    results = evaluate_baseline(
        model=model,
        dataset=dataset,
        vocab=vocab,
        num_batches=args.num_batches,
        adversary_epochs=args.adversary_epochs
    )

    # Save results
    output_dir = args.output_dir or config.RESULTS_DIR
    save_results(results, output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
