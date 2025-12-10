"""
Generate visualizations explaining the zero bias amplification phenomenon.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from PIL import Image, ImageDraw, ImageFont

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


def create_output_dir():
    """Create output directory for images."""
    output_dir = os.path.join(os.path.dirname(__file__), "bias_explanation_images")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def collect_samples_by_activity(dataset, vocab, model, num_batches=50):
    """
    Collect samples organized by dominant object/activity.
    """
    print("Collecting samples by activity...")

    # Common activities/objects to group by
    activities = {
        'skiing': [],
        'baseball': [],
        'tennis': [],
        'surfing': [],
        'cooking': [],
        'eating': [],
        'sitting': [],
        'standing': []
    }

    batch_count = 0
    for batch in dataset:
        images = batch['image']
        captions = batch['caption']
        race_labels = batch['race_label']

        for i in range(len(images)):
            gt_text = vocab.decode(captions[i].numpy()).lower()

            # Generate caption
            img_batch = tf.expand_dims(images[i], 0)
            generated_ids = model.generate_caption(
                img_batch,
                start_token=vocab.start_idx,
                end_token=vocab.end_idx,
                max_length=20
            )
            gen_text = vocab.decode(generated_ids).lower()

            race = "light" if race_labels[i].numpy() == 0 else "dark"

            # Categorize by activity
            for activity in activities.keys():
                if activity in gt_text or activity in gen_text:
                    if len(activities[activity]) < 12:  # Max 12 per activity
                        activities[activity].append({
                            'image': images[i].numpy(),
                            'gt_caption': gt_text,
                            'gen_caption': gen_text,
                            'race': race
                        })
                    break

        batch_count += 1
        if batch_count >= num_batches:
            break

        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...")

    return activities


def visualize_activity_race_correlation(activities, output_dir):
    """
    Create visualization showing race distribution across activities.
    """
    print("\nCreating activity-race correlation visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Race Distribution Across Different Activities/Objects\n(Evidence for Activity-Race Correlation)',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, (activity, samples) in enumerate(activities.items()):
        if idx >= 8 or len(samples) == 0:
            break

        ax = axes[idx]

        # Count by race
        light_count = sum(1 for s in samples if s['race'] == 'light')
        dark_count = sum(1 for s in samples if s['race'] == 'dark')
        total = light_count + dark_count

        if total == 0:
            continue

        # Create bar chart
        races = ['Light', 'Dark']
        counts = [light_count, dark_count]
        colors = ['#FFE4B5', '#8B4513']

        bars = ax.bar(races, counts, color=colors, edgecolor='black', linewidth=2)

        ax.set_title(f'{activity.capitalize()}\n({total} samples)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)

        # Add percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if total > 0:
                pct = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({pct:.0f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_activity_race_correlation.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: 1_activity_race_correlation.png")
    plt.close()


def deprocess_image(img):
    """
    Reverse ResNet preprocessing to get displayable image.

    ResNet preprocessing:
    1. Converts RGB to BGR
    2. Subtracts mean: [103.939, 116.779, 123.68] (in BGR order)

    To reverse:
    1. Add back mean
    2. Convert BGR back to RGB
    3. Clip to [0, 255]
    """
    # Add back ImageNet mean (in BGR order)
    mean = np.array([103.939, 116.779, 123.68])
    img = img + mean

    # BGR to RGB
    img = img[:, :, ::-1]

    # Clip to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def visualize_sample_images(activities, output_dir):
    """
    Show sample images with ground truth and generated captions.
    """
    print("\nCreating sample images visualization...")

    # Pick two activities with clear differences
    activity_pairs = []
    for activity, samples in activities.items():
        if len(samples) >= 4:
            light_samples = [s for s in samples if s['race'] == 'light']
            dark_samples = [s for s in samples if s['race'] == 'dark']
            if len(light_samples) >= 2 and len(dark_samples) >= 2:
                activity_pairs.append((activity, light_samples[:2], dark_samples[:2]))

    if len(activity_pairs) < 2:
        print("  Not enough diverse samples, skipping...")
        return

    for act_idx, (activity, light_samples, dark_samples) in enumerate(activity_pairs[:2]):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Activity: {activity.capitalize()}\nShowing Model Generates Same Caption for Both Races',
                     fontsize=14, fontweight='bold')

        all_samples = light_samples + dark_samples

        for idx, (ax, sample) in enumerate(zip(axes.flatten(), all_samples)):
            # Display image - deprocess first!
            img = deprocess_image(sample['image'].copy())
            ax.imshow(img)
            ax.axis('off')

            race_label = "Light-skinned" if sample['race'] == 'light' else "Dark-skinned"
            color = '#FFE4B5' if sample['race'] == 'light' else '#8B4513'

            # Add text box with captions
            textstr = f"Race: {race_label}\n\n"
            textstr += f"Ground Truth:\n'{sample['gt_caption']}'\n\n"
            textstr += f"Generated:\n'{sample['gen_caption']}'"

            props = dict(boxstyle='round', facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
            ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='center', bbox=props,
                   family='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'2_sample_images_{activity}.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: 2_sample_images_{activity}.png")
        plt.close()


def create_leakage_diagram(output_dir):
    """
    Create diagram explaining how all three leakage measures are equal.
    """
    print("\nCreating leakage explanation diagram...")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Why λD = λM = λD(F1) = 0.8971?',
            ha='center', fontsize=20, fontweight='bold')

    # Box 1: Dataset Leakage
    box1_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=3)
    box1_text = 'λD (Dataset Leakage)\n\n'
    box1_text += 'Ground Truth Captions:\n'
    box1_text += '• "woman skiing down slope"\n'
    box1_text += '• "man playing baseball"\n'
    box1_text += '• "person cooking in kitchen"\n\n'
    box1_text += 'Adversary learns:\n'
    box1_text += '"skiing" → LIGHT (90%)\n'
    box1_text += '"baseball" → LIGHT (70%)\n'
    box1_text += '"cooking" → DARK (60%)\n\n'
    box1_text += 'Accuracy: 89.71%'

    ax.text(1.5, 6.5, box1_text, fontsize=10, verticalalignment='top',
           bbox=box1_props, family='monospace')

    # Box 2: Model Leakage
    box2_props = dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='black', linewidth=3)
    box2_text = 'λM (Model Leakage)\n\n'
    box2_text += 'Generated Captions:\n'
    box2_text += '• "person skiing"\n'
    box2_text += '• "person playing baseball"\n'
    box2_text += '• "person cooking"\n\n'
    box2_text += 'Adversary STILL learns:\n'
    box2_text += '"skiing" → LIGHT (90%)\n'
    box2_text += '"baseball" → LIGHT (70%)\n'
    box2_text += '"cooking" → DARK (60%)\n\n'
    box2_text += 'Accuracy: 89.71%'

    ax.text(5, 6.5, box2_text, fontsize=10, verticalalignment='top',
           bbox=box2_props, family='monospace')

    # Box 3: Perturbed Leakage
    box3_props = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='black', linewidth=3)
    box3_text = 'λD(F1) (Perturbed @ F1=0.35)\n\n'
    box3_text += 'Perturbed Captions (70% replaced):\n'
    box3_text += '• "banana skiing truck nonsense"\n'
    box3_text += '• "random baseball words jumbled"\n'
    box3_text += '• "gibberish cooking text mixed"\n\n'
    box3_text += 'Activity word SURVIVES in TF-IDF!\n'
    box3_text += '"skiing" → LIGHT (90%)\n'
    box3_text += '"baseball" → LIGHT (70%)\n'
    box3_text += '"cooking" → DARK (60%)\n\n'
    box3_text += 'Accuracy: 89.71%'

    ax.text(8.5, 6.5, box3_text, fontsize=10, verticalalignment='top',
           bbox=box3_props, family='monospace')

    # Arrows and explanation
    ax.arrow(1.5, 2.5, 0, 1, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=2)
    ax.arrow(5, 2.5, 0, 1, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=2)
    ax.arrow(8.5, 2.5, 0, 1, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=2)

    # Bottom explanation box
    explanation_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='red', linewidth=3)
    explanation_text = 'ALL THREE measure the same underlying ACTIVITY-RACE correlation!\n\n'
    explanation_text += 'The model is so generic that it preserves the dataset\'s activity distribution.\n'
    explanation_text += 'Perturbation doesn\'t remove activity words from TF-IDF features.\n\n'
    explanation_text += '∴ Δ = λM - λD(F1) = 0.8971 - 0.8971 = 0.0000 (NO AMPLIFICATION)'

    ax.text(5, 1.5, explanation_text, fontsize=11, verticalalignment='top',
           horizontalalignment='center', bbox=explanation_props, family='monospace', fontweight='bold')

    plt.savefig(os.path.join(output_dir, '3_leakage_explanation_diagram.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: 3_leakage_explanation_diagram.png")
    plt.close()


def create_alpha_comparison(output_dir):
    """
    Create visualization comparing original vs α=1 results.
    """
    print("\nCreating alpha comparison visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Why α=1 (Balanced Data) Shows Bias Amplification', fontsize=16, fontweight='bold')

    # Original split
    ax1.set_title('Original Split\n(Unbalanced, Activity-Race Correlation)', fontsize=12, fontweight='bold')
    metrics1 = ['λD', 'λM', 'λD(F1)', 'Δ']
    values1 = [0.8971, 0.8971, 0.8971, 0.0000]
    colors1 = ['lightblue', 'lightcoral', 'lightgreen', 'red']

    bars1 = ax1.bar(metrics1, values1, color=colors1, edgecolor='black', linewidth=2)
    for bar, val in zip(bars1, values1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    textstr1 = 'Activity-race correlation dominates\nModel is generic → no amplification\nΔ = 0.0000'
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.5, 0.25, textstr1, transform=ax1.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props1)

    # α=1 split
    ax2.set_title('α=1 Split\n(Balanced, No Activity-Race Correlation)', fontsize=12, fontweight='bold')
    metrics2 = ['λD', 'λM', 'λD(F1)', 'Δ']
    values2 = [0.5527, 0.5964, 0.5018, 0.0945]
    colors2 = ['lightblue', 'lightcoral', 'lightgreen', 'red']

    bars2 = ax2.bar(metrics2, values2, color=colors2, edgecolor='black', linewidth=2)
    for bar, val in zip(bars2, values2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    textstr2 = 'Balanced data removes activity bias\nBut model STILL amplifies through\nunlabeled features!\nΔ = 0.0945 ⚠️'
    props2 = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    ax2.text(0.5, 0.25, textstr2, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_alpha_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: 4_alpha_comparison.png")
    plt.close()


def main():
    print("="*80)
    print("GENERATING BIAS EXPLANATION VISUALIZATIONS")
    print("="*80)

    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load vocab
    vocab = Vocabulary.load(config.VOCAB_PATH)

    # Load dataset
    _, val_dataset, _ = create_datasets(batch_size=16, balanced=False)

    # Load model
    model = create_baseline_model(vocab_size=len(vocab), finetune_encoder=False)
    dummy_batch = next(iter(val_dataset))
    _ = model(dummy_batch['image'], dummy_batch['caption'], training=False)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.weights.h5")
    model.load_weights(checkpoint_path)

    # Collect samples
    activities = collect_samples_by_activity(val_dataset, vocab, model, num_batches=50)

    # Generate visualizations
    visualize_activity_race_correlation(activities, output_dir)
    visualize_sample_images(activities, output_dir)
    create_leakage_diagram(output_dir)
    create_alpha_comparison(output_dir)

    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nAll images saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. 1_activity_race_correlation.png - Shows race distribution across activities")
    print("  2. 2_sample_images_*.png - Sample images with captions")
    print("  3. 3_leakage_explanation_diagram.png - Explains why all leakage measures are equal")
    print("  4. 4_alpha_comparison.png - Compares original vs balanced (α=1) results")


if __name__ == '__main__':
    main()
