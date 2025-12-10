"""
Evaluate baseline model on multiple balanced splits.
Replicates Table 1 from "Balanced Datasets Are Not Enough" for race bias.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_datasets
from models.baseline_model import create_baseline_model
from evaluation.caption_metrics import evaluate_captions
from evaluation.bias_metrics import comprehensive_bias_evaluation
from utils.vocab import Vocabulary
import utils.config as config


def create_balanced_split(all_data: List, all_labels: List[int],
                         alpha: float = 1.0) -> Tuple[List, List[int], Dict]:
    """
    Create a balanced subset where light/dark ratio ≤ alpha.

    Args:
        all_data: List of all data samples (tuples of data elements)
        all_labels: Race labels (0=Light, 1=Dark)
        alpha: Balance constraint (1.0 = perfectly balanced)

    Returns:
        subset_data: Balanced subset
        subset_labels: Labels for subset
        stats: Statistics
    """
    # Separate by race
    light_data = [(d, l) for d, l in zip(all_data, all_labels) if l == 0]
    dark_data = [(d, l) for d, l in zip(all_data, all_labels) if l == 1]

    light_count = len(light_data)
    dark_count = len(dark_data)

    # Determine target counts
    if alpha == 1.0:
        # Perfectly balanced
        target = min(light_count, dark_count)
        light_target, dark_target = target, target
    else:
        # Allow ratio up to alpha
        if light_count > dark_count:
            dark_target = dark_count
            light_target = min(light_count, int(dark_count * alpha))
        else:
            light_target = light_count
            dark_target = min(dark_count, int(light_count * alpha))

    # Random sample
    np.random.seed(42)
    light_subset = [light_data[i] for i in np.random.choice(
        len(light_data), light_target, replace=False
    )]
    dark_subset = [dark_data[i] for i in np.random.choice(
        len(dark_data), dark_target, replace=False
    )]

    # Combine and shuffle
    subset = light_subset + dark_subset
    np.random.shuffle(subset)

    subset_data = [d for d, _ in subset]
    subset_labels = [l for _, l in subset]

    stats = {
        'light_count': light_target,
        'dark_count': dark_target,
        'total': light_target + dark_target,
        'light_ratio': light_target / (light_target + dark_target),
        'actual_ratio': light_target / dark_target if dark_target > 0 else float('inf')
    }

    return subset_data, subset_labels, stats


def collect_all_data(dataset, vocab, num_batches=None):
    """
    Collect all data from dataset for subsetting.

    Returns:
        all_images: List of images
        all_captions: List of caption text
        all_race_labels: List of race labels
        all_image_ids: List of image IDs
    """
    print("\nCollecting dataset...")

    all_images = []
    all_captions = []
    all_race_labels = []
    all_image_ids = []

    batch_count = 0
    for batch in dataset:
        images = batch['image']
        captions = batch['caption']
        race_labels = batch['race_label']
        image_ids = batch.get('image_id', None)

        for i in range(len(images)):
            all_images.append(images[i].numpy())
            all_captions.append(vocab.decode(captions[i].numpy()))
            all_race_labels.append(race_labels[i].numpy())
            if image_ids is not None:
                all_image_ids.append(image_ids[i].numpy())
            else:
                all_image_ids.append(i)  # Fallback to index if no image_id

        batch_count += 1
        if num_batches and batch_count >= num_batches:
            break

        if batch_count % 10 == 0:
            print(f"  Collected {batch_count} batches...")

    print(f"Total samples: {len(all_images)}")
    print(f"  Light: {sum(1 for l in all_race_labels if l == 0)}")
    print(f"  Dark: {sum(1 for l in all_race_labels if l == 1)}")

    return all_images, all_captions, all_race_labels, all_image_ids


def evaluate_on_split(model, images, ground_truth, race_labels, vocab, split_name, image_ids=None):
    """
    Evaluate model on a specific data split.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ON SPLIT: {split_name}")
    print(f"{'='*80}")

    # Generate captions
    print("Generating captions...")
    generated = []

    for i, img in enumerate(images):
        # Convert numpy back to tensor
        img_tensor = tf.constant(img)
        img_batch = tf.expand_dims(img_tensor, 0)

        # Generate
        generated_ids = model.generate_caption(
            img_batch,
            start_token=vocab.start_idx,
            end_token=vocab.end_idx,
            max_length=20
        )

        generated_text = vocab.decode(generated_ids)
        generated.append(generated_text)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{len(images)} captions...")

    # Dataset statistics
    light_count = sum(1 for l in race_labels if l == 0)
    dark_count = sum(1 for l in race_labels if l == 1)

    dataset_stats = {
        'light_count': int(light_count),
        'dark_count': int(dark_count)
    }

    # Run comprehensive bias evaluation
    results = comprehensive_bias_evaluation(
        generated_captions=generated,
        ground_truth_captions=ground_truth,
        race_labels=race_labels,
        vocab=vocab,
        dataset_stats=dataset_stats,
        image_ids=image_ids
    )

    # Add per-race quality
    light_gen = [g for g, l in zip(generated, race_labels) if l == 0]
    light_ref = [[gt] for gt, l in zip(ground_truth, race_labels) if l == 0]

    dark_gen = [g for g, l in zip(generated, race_labels) if l == 1]
    dark_ref = [[gt] for gt, l in zip(ground_truth, race_labels) if l == 1]

    if len(light_gen) > 0:
        light_metrics = evaluate_captions(light_gen, light_ref)
        results['light_quality'] = light_metrics

    if len(dark_gen) > 0:
        dark_metrics = evaluate_captions(dark_gen, dark_ref)
        results['dark_quality'] = dark_metrics

    if len(light_gen) > 0 and len(dark_gen) > 0:
        results['quality_gaps'] = {
            'bleu4_gap': float(light_metrics['BLEU-4'] - dark_metrics['BLEU-4']),
            'f1_gap': float(light_metrics['F1'] - dark_metrics['F1'])
        }

    return results


def format_multi_split_table(all_results: Dict[str, Dict]) -> str:
    """
    Format results from multiple splits into a table matching the paper's Table 1.
    """
    lines = []
    lines.append("=" * 120)
    lines.append("BIAS EVALUATION RESULTS - MULTIPLE SPLITS (Paper Table 1 Format)")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'Split':<20} {'Statistics':<25} {'Leakage':<35} {'Performance':<20}")
    lines.append(f"{'':<20} {'#light':<10} {'#dark':<10} {'λD':<10} {'λM(F1)':<10} {'λD(F1)':<10} {'Δ':<10} {'F1':<10}")
    lines.append("-" * 120)

    for split_name, results in all_results.items():
        stats = results['statistics']

        row = f"{split_name:<20} "
        row += f"{stats['#light']:<10} "
        row += f"{stats['#dark']:<10} "
        row += f"{results['lambda_D']:<10.4f} "
        row += f"{results['lambda_M']:<10.4f} "
        row += f"{results['lambda_D_F1']:<10.4f} "
        row += f"{results['bias_amplification']:<10.4f} "
        row += f"{results['F1']:<10.4f}"

        lines.append(row)

    lines.append("=" * 120)
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  λD      = Dataset Leakage (predictability of race from ground truth captions)")
    lines.append("  λM(F1)  = Model Leakage (predictability of race from generated captions)")
    lines.append("  λD(F1)  = Expected leakage by chance at model's F1")
    lines.append("  Δ       = Bias Amplification (λM - λD(F1))")
    lines.append("")
    lines.append("KEY FINDING:")
    lines.append("  If Δ remains high even with balanced data (α=1), then")
    lines.append("  'Balanced Datasets Are Not Enough' - bias persists regardless of balance!")
    lines.append("=" * 120)

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate baseline on multiple splits')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--use_val', action='store_true')
    parser.add_argument('--splits', type=str, default='original,alpha_2,alpha_1',
                       help='Comma-separated splits to evaluate (original,alpha_3,alpha_2,alpha_1)')

    args = parser.parse_args()

    # Parse splits
    requested_splits = args.splits.split(',')

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Vocabulary.load(config.VOCAB_PATH)

    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = create_datasets(
        batch_size=args.batch_size,
        balanced=False
    )

    dataset = val_dataset if args.use_val else train_dataset

    # Create model
    print("\nCreating model...")
    model = create_baseline_model(vocab_size=len(vocab), finetune_encoder=False)

    # Build model
    print("Building model...")
    dummy_batch = next(iter(dataset))
    _ = model(dummy_batch['image'], dummy_batch['caption'], training=False)

    # Load weights
    print(f"Loading weights from {args.checkpoint}...")
    model.load_weights(args.checkpoint)
    print("Weights loaded!")

    # Collect all data
    all_images, all_captions, all_race_labels, all_image_ids = collect_all_data(
        dataset, vocab, args.num_batches
    )

    # Evaluate on different splits
    all_results = {}

    for split_name in requested_splits:
        if split_name == 'original':
            # Use all data
            images = all_images
            captions = all_captions
            labels = all_race_labels
            img_ids = all_image_ids
            display_name = 'original'

        elif split_name.startswith('alpha_'):
            # Create balanced subset
            alpha = float(split_name.split('_')[1])

            # Combine data for splitting
            combined = list(zip(all_images, all_captions, all_image_ids, all_race_labels))

            # Create balanced split
            subset, subset_labels, split_stats = create_balanced_split(
                [(img, cap, img_id) for img, cap, img_id, _ in combined],
                all_race_labels,
                alpha=alpha
            )

            images = [img for img, _, _ in subset]
            captions = [cap for _, cap, _ in subset]
            img_ids = [img_id for _, _, img_id in subset]
            labels = subset_labels
            display_name = f'α = {alpha}'

            print(f"\nBalanced split (α={alpha}):")
            print(f"  Light: {split_stats['light_count']}")
            print(f"  Dark: {split_stats['dark_count']}")
            print(f"  Ratio: {split_stats['actual_ratio']:.2f}:1")

        else:
            print(f"Unknown split: {split_name}, skipping...")
            continue

        # Evaluate
        results = evaluate_on_split(
            model, images, captions, labels, vocab, display_name, image_ids=img_ids
        )
        all_results[display_name] = results

    # Save results
    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_file = os.path.join(output_dir, f'baseline_all_splits_{timestamp}.json')
    with open(json_file, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print(f"\nJSON saved to: {json_file}")

    # Save formatted table
    table_file = os.path.join(output_dir, f'baseline_table_all_splits_{timestamp}.txt')
    table_str = format_multi_split_table(all_results)

    with open(table_file, 'w') as f:
        f.write(table_str)
        f.write("\n\n")

        # Add detailed metrics
        f.write("="*120 + "\n")
        f.write("DETAILED METRICS BY SPLIT\n")
        f.write("="*120 + "\n\n")

        for split_name, results in all_results.items():
            f.write(f"\n{split_name}:\n")
            f.write("-" * 60 + "\n")

            if 'caption_quality' in results:
                f.write("Overall Caption Quality:\n")
                for key, value in results['caption_quality'].items():
                    f.write(f"  {key}: {value:.4f}\n")

            if 'light_quality' in results and 'dark_quality' in results:
                f.write("\nPer-Race Quality:\n")
                f.write(f"  Light BLEU-4: {results['light_quality']['BLEU-4']:.4f}\n")
                f.write(f"  Dark BLEU-4:  {results['dark_quality']['BLEU-4']:.4f}\n")
                if 'quality_gaps' in results:
                    f.write(f"  Gap: {results['quality_gaps']['bleu4_gap']:+.4f}\n")

    print(f"Table saved to: {table_file}")

    # Print table
    print("\n" + table_str)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
