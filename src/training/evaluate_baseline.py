"""
Corrected evaluation script using paper-correct bias metrics.
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
from evaluation.caption_metrics import evaluate_captions
from evaluation.bias_metrics import comprehensive_bias_evaluation, format_results_table
from utils.vocab import Vocabulary
import utils.config as config


def evaluate_baseline(model, dataset, vocab, num_batches=None):
    """
    Evaluate baseline model using CORRECTED paper methodology.

    Args:
        model: Trained baseline model
        dataset: Evaluation dataset
        vocab: Vocabulary
        num_batches: Number of batches to evaluate (None = all)

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (CORRECTED METHODOLOGY)")
    print("=" * 80)

    # Collect data
    all_generated = []
    all_ground_truth = []
    all_race_labels = []

    print("\nGenerating captions...")

    batch_count = 0
    for batch in dataset:
        images = batch['image']
        captions = batch['caption']
        race_labels = batch['race_label']

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
            ground_truth_text = vocab.decode(captions[i].numpy())

            all_generated.append(generated_text)
            all_ground_truth.append(ground_truth_text)

        all_race_labels.extend(race_labels.numpy().tolist())

        batch_count += 1
        if num_batches and batch_count >= num_batches:
            break

        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...")

    print(f"\nTotal samples evaluated: {len(all_generated)}")

    # Dataset statistics
    race_labels_array = np.array(all_race_labels)
    light_count = np.sum(race_labels_array == 0)
    dark_count = np.sum(race_labels_array == 1)

    dataset_stats = {
        'light_count': int(light_count),
        'dark_count': int(dark_count)
    }

    # Run comprehensive bias evaluation using paper methodology
    results = comprehensive_bias_evaluation(
        generated_captions=all_generated,
        ground_truth_captions=all_ground_truth,
        race_labels=all_race_labels,
        vocab=vocab,
        dataset_stats=dataset_stats
    )

    # Add per-race quality analysis
    print(f"\n{'='*80}")
    print("PER-RACE CAPTION QUALITY")
    print(f"{'='*80}")

    light_gen = [cap for cap, race in zip(all_generated, all_race_labels) if race == 0]
    light_ref = [[gt] for gt, race in zip(all_ground_truth, all_race_labels) if race == 0]

    dark_gen = [cap for cap, race in zip(all_generated, all_race_labels) if race == 1]
    dark_ref = [[gt] for gt, race in zip(all_ground_truth, all_race_labels) if race == 1]

    if len(light_gen) > 0:
        light_metrics = evaluate_captions(light_gen, light_ref)
        results['light_quality'] = light_metrics
        print(f"\nLight skin samples ({len(light_gen)}):")
        print(f"  BLEU-4: {light_metrics['BLEU-4']:.4f}")
        print(f"  F1:     {light_metrics['F1']:.4f}")

    if len(dark_gen) > 0:
        dark_metrics = evaluate_captions(dark_gen, dark_ref)
        results['dark_quality'] = dark_metrics
        print(f"\nDark skin samples ({len(dark_gen)}):")
        print(f"  BLEU-4: {dark_metrics['BLEU-4']:.4f}")
        print(f"  F1:     {dark_metrics['F1']:.4f}")

    if len(light_gen) > 0 and len(dark_gen) > 0:
        bleu4_gap = light_metrics['BLEU-4'] - dark_metrics['BLEU-4']
        f1_gap = light_metrics['F1'] - dark_metrics['F1']
        results['quality_gaps'] = {
            'bleu4_gap': float(bleu4_gap),
            'f1_gap': float(f1_gap)
        }
        print(f"\nQuality Gaps:")
        print(f"  BLEU-4 Gap: {bleu4_gap:+.4f} {'(favors light)' if bleu4_gap > 0 else '(favors dark)'}")
        print(f"  F1 Gap:     {f1_gap:+.4f} {'(favors light)' if f1_gap > 0 else '(favors dark)'}")

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

    # Save JSON
    output_file = os.path.join(output_dir, f'baseline_eval_corrected_{timestamp}.json')

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

    # Save formatted table
    table_file = os.path.join(output_dir, f'baseline_table_corrected_{timestamp}.txt')
    table_str = format_results_table(results, split_name="baseline")

    with open(table_file, 'w') as f:
        f.write(table_str)
        f.write("\n\n")

        # Add detailed metrics
        f.write("="*100 + "\n")
        f.write("DETAILED CAPTION QUALITY METRICS\n")
        f.write("="*100 + "\n\n")

        if 'caption_quality' in results:
            f.write("Overall:\n")
            for key, value in results['caption_quality'].items():
                f.write(f"  {key}: {value:.4f}\n")

        if 'light_quality' in results:
            f.write("\nLight skin:\n")
            for key, value in results['light_quality'].items():
                f.write(f"  {key}: {value:.4f}\n")

        if 'dark_quality' in results:
            f.write("\nDark skin:\n")
            for key, value in results['dark_quality'].items():
                f.write(f"  {key}: {value:.4f}\n")

        if 'quality_gaps' in results:
            f.write("\nQuality Gaps:\n")
            for key, value in results['quality_gaps'].items():
                f.write(f"  {key}: {value:+.4f}\n")

    print(f"Table saved to: {table_file}")

    # Also print to console
    print("\n" + table_str)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate baseline model (CORRECTED)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint or weights')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Number of batches to evaluate (None = all)')
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

    # Build model by calling it on a dummy batch
    print("Building model...")
    dummy_batch = next(iter(dataset))
    _ = model(dummy_batch['image'], dummy_batch['caption'], training=False)
    print("Model built successfully!")

    # Load weights
    print(f"Loading weights from {args.checkpoint}...")
    model.load_weights(args.checkpoint)
    print("Weights loaded successfully!")

    # Evaluate
    results = evaluate_baseline(
        model=model,
        dataset=dataset,
        vocab=vocab,
        num_batches=args.num_batches
    )

    # Save results
    output_dir = args.output_dir or config.RESULTS_DIR
    save_results(results, output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
