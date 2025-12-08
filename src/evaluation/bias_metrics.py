"""
Bias evaluation metrics for image captioning models.
Measures racial bias through adversary probing, word frequency, and fairness metrics.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adversary import RaceAdversary
from evaluation.caption_metrics import CaptionMetrics
import utils.config as config


# List of racial descriptors to monitor
RACIAL_DESCRIPTORS = [
    'black', 'white', 'dark', 'light', 'african', 'asian', 'hispanic', 'latino',
    'caucasian', 'colored', 'brown', 'pale', 'ethnic', 'race', 'racial',
    'skin', 'tone'
]


class BiasMetrics:
    """
    Comprehensive bias evaluation metrics.
    """

    def __init__(self):
        """Initialize bias metrics."""
        self.caption_metrics = CaptionMetrics()

    def train_adversary_probe(self, features, race_labels, input_dim,
                             epochs=20, batch_size=32, learning_rate=1e-3):
        """
        Train adversary to predict race from frozen features.
        Higher accuracy = more racial information in features (more biased).

        Args:
            features: Model features [N, feature_dim]
            race_labels: Race labels [N] (0=Light, 1=Dark)
            input_dim: Feature dimension
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dictionary with adversary accuracy and loss
        """
        print(f"\nTraining adversary probe on {len(features)} samples...")

        # Create adversary
        adversary = RaceAdversary(
            input_dim=input_dim,
            hidden_dims=[512, 512, 512] if input_dim > 1000 else [256, 256],
            num_classes=2,
            dropout=0.3
        )

        # Loss and optimizer
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, race_labels))
        dataset = dataset.shuffle(10000).batch(batch_size)

        # Training loop
        best_accuracy = 0.0

        for epoch in range(epochs):
            epoch_loss = []
            epoch_accuracy = []

            for batch_features, batch_labels in dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    logits = adversary(batch_features, training=True)
                    loss = loss_fn(batch_labels, logits)

                # Backward pass
                gradients = tape.gradient(loss, adversary.trainable_variables)
                optimizer.apply_gradients(zip(gradients, adversary.trainable_variables))

                # Compute accuracy
                preds = tf.argmax(logits, axis=-1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(preds, batch_labels), tf.float32)
                )

                epoch_loss.append(loss.numpy())
                epoch_accuracy.append(accuracy.numpy())

            avg_loss = np.mean(epoch_loss)
            avg_accuracy = np.mean(epoch_accuracy)

            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

        print(f"\nAdversary probe results:")
        print(f"  Best accuracy: {best_accuracy:.4f}")
        print(f"  Interpretation: {'BIASED' if best_accuracy > 0.6 else 'Less biased'} "
              f"({'features contain racial info' if best_accuracy > 0.6 else 'features more race-invariant'})")

        return {
            'adversary_accuracy': best_accuracy,
            'final_loss': avg_loss
        }

    def compute_race_word_frequency(self, captions: List[str],
                                   race_labels: List[int]) -> Dict:
        """
        Compute frequency of racial descriptors in captions.

        Args:
            captions: List of generated captions
            race_labels: Race labels for each caption

        Returns:
            Dictionary with racial word statistics
        """
        light_words = []
        dark_words = []

        for caption, race in zip(captions, race_labels):
            words = caption.lower().split()
            if race == 0:  # Light
                light_words.extend(words)
            else:  # Dark
                dark_words.extend(words)

        # Count racial descriptors
        light_racial_count = sum(1 for word in light_words if word in RACIAL_DESCRIPTORS)
        dark_racial_count = sum(1 for word in dark_words if word in RACIAL_DESCRIPTORS)

        light_total = len(light_words)
        dark_total = len(dark_words)

        results = {
            'light_racial_word_freq': light_racial_count / light_total if light_total > 0 else 0,
            'dark_racial_word_freq': dark_racial_count / dark_total if dark_total > 0 else 0,
            'light_racial_count': light_racial_count,
            'dark_racial_count': dark_racial_count,
            'racial_word_disparity': abs(
                (light_racial_count / light_total if light_total > 0 else 0) -
                (dark_racial_count / dark_total if dark_total > 0 else 0)
            )
        }

        return results

    def compute_per_race_quality(self, generated_captions: List[str],
                                reference_captions: List[List[str]],
                                race_labels: List[int]) -> Dict:
        """
        Compute caption quality separately for each race group.

        Args:
            generated_captions: Generated captions
            reference_captions: Reference captions
            race_labels: Race labels

        Returns:
            Dictionary with per-race BLEU scores
        """
        # Separate by race
        light_gen = [cap for cap, race in zip(generated_captions, race_labels) if race == 0]
        light_ref = [ref for ref, race in zip(reference_captions, race_labels) if race == 0]

        dark_gen = [cap for cap, race in zip(generated_captions, race_labels) if race == 1]
        dark_ref = [ref for ref, race in zip(reference_captions, race_labels) if race == 1]

        results = {}

        # Compute metrics for Light
        if len(light_gen) > 0:
            light_gen_tokens = [cap.lower().split() for cap in light_gen]
            light_ref_tokens = [[ref.lower().split() for ref in refs] for refs in light_ref]

            light_metrics = self.caption_metrics.compute_all_metrics(
                light_ref_tokens, light_gen_tokens
            )
            results['light'] = light_metrics
            results['light_count'] = len(light_gen)

        # Compute metrics for Dark
        if len(dark_gen) > 0:
            dark_gen_tokens = [cap.lower().split() for cap in dark_gen]
            dark_ref_tokens = [[ref.lower().split() for ref in refs] for refs in dark_ref]

            dark_metrics = self.caption_metrics.compute_all_metrics(
                dark_ref_tokens, dark_gen_tokens
            )
            results['dark'] = dark_metrics
            results['dark_count'] = len(dark_gen)

        # Compute disparity
        if 'light' in results and 'dark' in results:
            results['bleu4_disparity'] = abs(
                results['light']['BLEU-4'] - results['dark']['BLEU-4']
            )
            results['quality_gap'] = (
                results['light']['BLEU-4'] - results['dark']['BLEU-4']
            )

        return results

    def compute_bias_amplification(self, model_race_accuracy: float,
                                  dataset_race_distribution: Dict) -> float:
        """
        Compute bias amplification index.
        Δ = (Model bias - Dataset bias) / Dataset bias

        Args:
            model_race_accuracy: Adversary accuracy on model features
            dataset_race_distribution: Dataset race distribution

        Returns:
            Bias amplification index
        """
        # Dataset bias: deviation from 50/50
        light_ratio = dataset_race_distribution.get('light', 0.5)
        dataset_bias = abs(light_ratio - 0.5)

        # Model bias: deviation from random (50% accuracy)
        model_bias = abs(model_race_accuracy - 0.5)

        # Amplification
        if dataset_bias > 0:
            amplification = (model_bias - dataset_bias) / dataset_bias
        else:
            amplification = 0.0

        return amplification


def evaluate_model_bias(model, dataset, vocab, adversary_epochs=20):
    """
    Comprehensive bias evaluation for a trained model.

    Args:
        model: Trained image captioning model
        dataset: tf.data.Dataset to evaluate on
        vocab: Vocabulary object
        adversary_epochs: Epochs to train adversary probe

    Returns:
        Dictionary with all bias metrics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE BIAS EVALUATION")
    print("="*60)

    # Collect features, captions, and labels
    all_encoder_features = []
    all_decoder_hiddens = []
    all_generated_captions = []
    all_reference_captions = []
    all_race_labels = []

    print("\nGenerating captions and extracting features...")

    for batch in dataset.take(100):  # Evaluate on subset for speed
        images = batch['image']
        captions = batch['caption']
        race_labels = batch['race_label']

        # Forward pass
        predictions, encoder_features, decoder_hiddens = model(
            images, captions, training=False
        )

        # Store encoder features
        all_encoder_features.append(encoder_features.numpy())

        # Store decoder hiddens (average across timesteps)
        decoder_avg = tf.reduce_mean(decoder_hiddens, axis=1)
        all_decoder_hiddens.append(decoder_avg.numpy())

        # Generate captions
        for i in range(len(images)):
            single_image = tf.expand_dims(images[i], 0)
            generated = model.generate_caption(
                single_image,
                start_token=vocab.start_idx,
                end_token=vocab.end_idx,
                max_length=20
            )
            caption_text = vocab.decode(generated)
            all_generated_captions.append(caption_text)

            # Reference caption
            ref_caption = vocab.decode(captions[i].numpy())
            all_reference_captions.append([ref_caption])

        all_race_labels.extend(race_labels.numpy().tolist())

    # Concatenate features
    encoder_features = np.concatenate(all_encoder_features, axis=0)
    decoder_features = np.concatenate(all_decoder_hiddens, axis=0)
    race_labels_array = np.array(all_race_labels)

    # Initialize metrics
    bias_metrics = BiasMetrics()
    results = {}

    # 1. Adversary probing on encoder features
    print("\n1. Adversary Probing on Encoder Features")
    print("-" * 60)
    encoder_adv_results = bias_metrics.train_adversary_probe(
        encoder_features,
        race_labels_array,
        input_dim=encoder_features.shape[1],
        epochs=adversary_epochs
    )
    results['encoder_adversary'] = encoder_adv_results

    # 2. Adversary probing on decoder features
    print("\n2. Adversary Probing on Decoder Features")
    print("-" * 60)
    decoder_adv_results = bias_metrics.train_adversary_probe(
        decoder_features,
        race_labels_array,
        input_dim=decoder_features.shape[1],
        epochs=adversary_epochs
    )
    results['decoder_adversary'] = decoder_adv_results

    # 3. Race word frequency
    print("\n3. Racial Descriptor Frequency")
    print("-" * 60)
    race_word_results = bias_metrics.compute_race_word_frequency(
        all_generated_captions,
        all_race_labels
    )
    results['race_words'] = race_word_results

    print(f"  Light skin - racial word frequency: {race_word_results['light_racial_word_freq']:.4f}")
    print(f"  Dark skin - racial word frequency: {race_word_results['dark_racial_word_freq']:.4f}")
    print(f"  Disparity: {race_word_results['racial_word_disparity']:.4f}")

    # 4. Per-race caption quality
    print("\n4. Per-Race Caption Quality")
    print("-" * 60)
    per_race_results = bias_metrics.compute_per_race_quality(
        all_generated_captions,
        all_reference_captions,
        all_race_labels
    )
    results['per_race_quality'] = per_race_results

    if 'light' in per_race_results:
        print(f"  Light skin - BLEU-4: {per_race_results['light']['BLEU-4']:.4f}")
    if 'dark' in per_race_results:
        print(f"  Dark skin - BLEU-4: {per_race_results['dark']['BLEU-4']:.4f}")
    if 'bleu4_disparity' in per_race_results:
        print(f"  Quality disparity: {per_race_results['bleu4_disparity']:.4f}")

    # 5. Dataset statistics
    light_count = np.sum(race_labels_array == 0)
    dark_count = np.sum(race_labels_array == 1)
    total = len(race_labels_array)

    results['dataset_stats'] = {
        'light_ratio': light_count / total,
        'dark_ratio': dark_count / total,
        'light_count': int(light_count),
        'dark_count': int(dark_count),
        'total': total
    }

    print("\n5. Dataset Statistics")
    print("-" * 60)
    print(f"  Light samples: {light_count} ({light_count/total*100:.1f}%)")
    print(f"  Dark samples: {dark_count} ({dark_count/total*100:.1f}%)")

    # 6. Bias amplification
    bias_amp = bias_metrics.compute_bias_amplification(
        encoder_adv_results['adversary_accuracy'],
        {'light': light_count / total}
    )
    results['bias_amplification'] = bias_amp

    print("\n6. Bias Amplification")
    print("-" * 60)
    print(f"  Amplification index: {bias_amp:.4f}")

    return results


def print_bias_summary(results: Dict, title: str = "Bias Evaluation Summary"):
    """
    Print comprehensive bias evaluation summary.

    Args:
        results: Dictionary from evaluate_model_bias
        title: Title to print
    """
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)

    # Adversary results
    print("\n📊 ADVERSARY PROBING (Higher = More Biased)")
    print("-" * 80)
    if 'encoder_adversary' in results:
        acc = results['encoder_adversary']['adversary_accuracy']
        status = "🔴 BIASED" if acc > 0.6 else "🟡 MODERATE" if acc > 0.55 else "🟢 DEBIASED"
        print(f"  Encoder Features Adversary Accuracy: {acc:.4f} {status}")

    if 'decoder_adversary' in results:
        acc = results['decoder_adversary']['adversary_accuracy']
        status = "🔴 BIASED" if acc > 0.6 else "🟡 MODERATE" if acc > 0.55 else "🟢 DEBIASED"
        print(f"  Decoder Features Adversary Accuracy: {acc:.4f} {status}")

    # Quality disparity
    print("\n📏 FAIRNESS METRICS")
    print("-" * 80)
    if 'per_race_quality' in results:
        pr = results['per_race_quality']
        if 'light' in pr and 'dark' in pr:
            print(f"  Light Skin BLEU-4: {pr['light']['BLEU-4']:.4f}")
            print(f"  Dark Skin BLEU-4: {pr['dark']['BLEU-4']:.4f}")
            if 'quality_gap' in pr:
                gap = pr['quality_gap']
                status = "🔴 UNFAIR" if abs(gap) > 0.02 else "🟢 FAIR"
                print(f"  Quality Gap: {gap:+.4f} {status}")

    # Race words
    print("\n📝 RACIAL DESCRIPTOR USAGE")
    print("-" * 80)
    if 'race_words' in results:
        rw = results['race_words']
        print(f"  Light Skin Captions: {rw['light_racial_word_freq']:.4f}")
        print(f"  Dark Skin Captions: {rw['dark_racial_word_freq']:.4f}")
        print(f"  Disparity: {rw['racial_word_disparity']:.4f}")

    # Overall verdict
    print("\n⚖️  OVERALL BIAS VERDICT")
    print("-" * 80)

    enc_acc = results.get('encoder_adversary', {}).get('adversary_accuracy', 0.5)

    if enc_acc > 0.65:
        verdict = "🔴 HIGHLY BIASED - Features strongly encode racial information"
    elif enc_acc > 0.6:
        verdict = "🟠 MODERATELY BIASED - Features contain racial signals"
    elif enc_acc > 0.55:
        verdict = "🟡 SLIGHTLY BIASED - Some racial leakage present"
    else:
        verdict = "🟢 DEBIASED - Features are largely race-invariant"

    print(f"  {verdict}")
    print("="*80)


if __name__ == '__main__':
    # Test bias metrics
    print("Testing Bias Metrics...\n")

    # Create dummy data
    features = np.random.randn(100, 2048).astype(np.float32)
    race_labels = np.random.randint(0, 2, size=100).astype(np.int32)

    bias_metrics = BiasMetrics()

    # Test adversary probing
    print("Testing adversary probing...")
    adv_results = bias_metrics.train_adversary_probe(
        features, race_labels, input_dim=2048, epochs=5
    )

    print(f"\nAdversary Results: {adv_results}")

    print("\n✅ Bias metrics test passed!")
