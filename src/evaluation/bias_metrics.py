"""
Corrected bias evaluation metrics matching the paper:
'Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations'

Key concepts:
- Dataset Leakage (λD): Adversary accuracy on GROUND TRUTH captions
- Model Leakage (λM): Adversary accuracy on MODEL-GENERATED captions
- Dataset Leakage at F1 (λD(F1)): Adversary accuracy on perturbed ground truth at model's F1
- Bias Amplification (Δ): λM(F1) - λD(F1)
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.caption_metrics import CaptionMetrics


class LeakageMetrics:
    """
    Implements leakage and bias amplification metrics from the paper.
    """

    def __init__(self, vocab_size=5000):
        """
        Initialize leakage metrics.

        Args:
            vocab_size: Vocabulary size for TF-IDF
        """
        self.caption_metrics = CaptionMetrics()
        self.vocab_size = vocab_size

    def compute_caption_features(self, captions: List[str], fit=True,
                                 vectorizer=None) -> Tuple[np.ndarray, object]:
        """
        Convert captions to TF-IDF features for adversary training.

        Args:
            captions: List of caption strings
            fit: Whether to fit vectorizer (True for train, False for test)
            vectorizer: Existing vectorizer (if fit=False)

        Returns:
            features: TF-IDF features [N, vocab_size]
            vectorizer: Fitted vectorizer
        """
        if fit or vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                stop_words='english',
                ngram_range=(1, 2)  # Unigrams and bigrams
            )
            features = vectorizer.fit_transform(captions).toarray()
        else:
            features = vectorizer.transform(captions).toarray()

        return features.astype(np.float32), vectorizer

    def train_adversary(self, caption_features: np.ndarray, race_labels: np.ndarray,
                       image_ids: np.ndarray = None, test_size=0.2, random_state=42) -> Dict:
        """
        Train logistic regression adversary to predict race from caption features.

        Args:
            caption_features: Caption TF-IDF features [N, vocab_size]
            race_labels: Race labels [N] (0=Light, 1=Dark)
            image_ids: Optional image IDs [N] - if provided, splits by image to prevent leakage
            test_size: Fraction for test split
            random_state: Random seed

        Returns:
            Dictionary with adversary results
        """
        if image_ids is not None:
            # Split by image ID to prevent data leakage
            # Get unique image IDs with their race labels
            unique_ids = np.unique(image_ids)
            id_to_race = {}
            for img_id, race in zip(image_ids, race_labels):
                if img_id not in id_to_race:
                    id_to_race[img_id] = race

            # Get race labels for unique images
            unique_races = np.array([id_to_race[img_id] for img_id in unique_ids])

            # Split unique image IDs
            train_ids, test_ids = train_test_split(
                unique_ids,
                test_size=test_size,
                random_state=random_state,
                stratify=unique_races
            )

            # Create train/test masks based on image IDs
            train_mask = np.isin(image_ids, train_ids)
            test_mask = np.isin(image_ids, test_ids)

            X_train = caption_features[train_mask]
            X_test = caption_features[test_mask]
            y_train = race_labels[train_mask]
            y_test = race_labels[test_mask]

            print(f"  Image-level split: {len(train_ids)} train images, {len(test_ids)} test images")
            print(f"  Caption-level: {len(X_train)} train captions, {len(X_test)} test captions")
        else:
            # Original caption-level split (WARNING: may have data leakage!)
            print(f"  WARNING: Splitting by captions, not images - may have data leakage!")
            X_train, X_test, y_train, y_test = train_test_split(
                caption_features, race_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=race_labels
            )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classifier': clf
        }

    def compute_dataset_leakage(self, ground_truth_captions: List[str],
                               race_labels: List[int], image_ids: List = None) -> Dict:
        """
        Compute dataset leakage: λD
        How well can we predict race from GROUND TRUTH captions?

        Args:
            ground_truth_captions: Ground truth caption strings
            race_labels: Race labels
            image_ids: Optional image IDs to prevent data leakage

        Returns:
            Dictionary with dataset leakage
        """
        print(f"\n{'='*80}")
        print("COMPUTING DATASET LEAKAGE (λD)")
        print(f"{'='*80}")
        print(f"Samples: {len(ground_truth_captions)}")

        # Convert captions to features
        features, vectorizer = self.compute_caption_features(
            ground_truth_captions, fit=True
        )

        # Train adversary
        image_ids_arr = np.array(image_ids) if image_ids is not None else None
        results = self.train_adversary(features, np.array(race_labels), image_ids=image_ids_arr)

        print(f"Dataset Leakage (λD): {results['test_accuracy']:.4f}")
        print(f"  → {'HIGH LEAKAGE' if results['test_accuracy'] > 0.6 else 'Moderate leakage'}")

        return {
            'lambda_D': results['test_accuracy'],
            'train_accuracy': results['train_accuracy'],
            'vectorizer': vectorizer,
            'classifier': results['classifier']
        }

    def compute_model_leakage(self, generated_captions: List[str],
                             race_labels: List[int],
                             vectorizer=None, image_ids: List = None) -> Dict:
        """
        Compute model leakage: λM
        How well can we predict race from MODEL-GENERATED captions?

        Args:
            generated_captions: Model-generated caption strings
            race_labels: Race labels
            vectorizer: Pre-fitted vectorizer (from dataset leakage)
            image_ids: Optional image IDs to prevent data leakage

        Returns:
            Dictionary with model leakage
        """
        print(f"\n{'='*80}")
        print("COMPUTING MODEL LEAKAGE (λM)")
        print(f"{'='*80}")
        print(f"Samples: {len(generated_captions)}")

        # Convert captions to features
        if vectorizer is None:
            features, vectorizer = self.compute_caption_features(
                generated_captions, fit=True
            )
        else:
            # Use same vectorizer as dataset for fair comparison
            features, _ = self.compute_caption_features(
                generated_captions, fit=False, vectorizer=vectorizer
            )

        # Train adversary
        image_ids_arr = np.array(image_ids) if image_ids is not None else None
        results = self.train_adversary(features, np.array(race_labels), image_ids=image_ids_arr)

        print(f"Model Leakage (λM): {results['test_accuracy']:.4f}")
        print(f"  → {'HIGH LEAKAGE' if results['test_accuracy'] > 0.6 else 'Moderate leakage'}")

        return {
            'lambda_M': results['test_accuracy'],
            'train_accuracy': results['train_accuracy'],
            'vectorizer': vectorizer,
            'classifier': results['classifier']
        }

    def perturb_captions(self, captions: List[str], references: List[List[str]],
                        target_f1: float, vocab) -> List[str]:
        """
        Randomly perturb captions to achieve target F1 score.
        Used to compute λD(F1) - dataset leakage at model's performance level.

        Args:
            captions: Ground truth captions
            references: Reference captions (same as captions for ground truth)
            target_f1: Target F1 score to achieve
            vocab: Vocabulary object

        Returns:
            Perturbed captions
        """
        print(f"\nPerturbing captions to achieve F1={target_f1:.4f}...")

        perturbed = captions.copy()
        current_f1 = 1.0
        perturbation_rate = 0.0

        # Binary search for perturbation rate
        low, high = 0.0, 1.0
        max_iterations = 20

        for iteration in range(max_iterations):
            perturbation_rate = (low + high) / 2.0

            # Perturb captions
            perturbed = []
            for caption in captions:
                words = caption.split()
                if len(words) == 0:
                    perturbed.append(caption)
                    continue

                # Randomly replace words
                new_words = []
                for word in words:
                    if np.random.random() < perturbation_rate:
                        # Replace with random word from vocab
                        random_idx = np.random.randint(0, len(vocab))
                        random_word = vocab.idx2word.get(random_idx, '<UNK>')
                        new_words.append(random_word)
                    else:
                        new_words.append(word)

                perturbed.append(' '.join(new_words))

            # Compute F1
            perturbed_tokens = [p.lower().split() for p in perturbed]
            ref_tokens = [[r.lower().split() for r in refs] for refs in references]
            metrics = self.caption_metrics.compute_all_metrics(ref_tokens, perturbed_tokens)
            current_f1 = metrics['F1']

            # Update binary search bounds
            if abs(current_f1 - target_f1) < 0.01:
                break
            elif current_f1 > target_f1:
                low = perturbation_rate
            else:
                high = perturbation_rate

            if iteration % 5 == 0:
                print(f"  Iteration {iteration}: F1={current_f1:.4f}, rate={perturbation_rate:.3f}")

        print(f"Final: F1={current_f1:.4f} (target={target_f1:.4f}), rate={perturbation_rate:.3f}")

        return perturbed

    def compute_dataset_leakage_at_f1(self, ground_truth_captions: List[str],
                                     references: List[List[str]],
                                     race_labels: List[int],
                                     model_f1: float,
                                     vocab,
                                     vectorizer=None, image_ids: List = None) -> Dict:
        """
        Compute dataset leakage at model's F1 level: λD(F1)
        Perturb ground truth to match model's F1, then measure leakage.

        Args:
            ground_truth_captions: Ground truth captions
            references: Reference captions
            race_labels: Race labels
            model_f1: Model's F1 score
            vocab: Vocabulary object
            vectorizer: Pre-fitted vectorizer
            image_ids: Optional image IDs to prevent data leakage

        Returns:
            Dictionary with λD(F1)
        """
        print(f"\n{'='*80}")
        print(f"COMPUTING DATASET LEAKAGE AT F1={model_f1:.4f} (λD(F1))")
        print(f"{'='*80}")

        # Perturb captions to match model F1
        perturbed_captions = self.perturb_captions(
            ground_truth_captions, references, model_f1, vocab
        )

        # Convert to features
        if vectorizer is None:
            features, vectorizer = self.compute_caption_features(
                perturbed_captions, fit=True
            )
        else:
            features, _ = self.compute_caption_features(
                perturbed_captions, fit=False, vectorizer=vectorizer
            )

        # Train adversary
        image_ids_arr = np.array(image_ids) if image_ids is not None else None
        results = self.train_adversary(features, np.array(race_labels), image_ids=image_ids_arr)

        print(f"Dataset Leakage at F1 (λD(F1)): {results['test_accuracy']:.4f}")

        return {
            'lambda_D_F1': results['test_accuracy'],
            'perturbed_captions': perturbed_captions
        }

    def compute_bias_amplification(self, lambda_M: float, lambda_D_F1: float) -> float:
        """
        Compute bias amplification: Δ = λM(F1) - λD(F1)

        Model amplifies bias if Δ > 0.

        Args:
            lambda_M: Model leakage
            lambda_D_F1: Dataset leakage at model's F1

        Returns:
            Bias amplification
        """
        delta = lambda_M - lambda_D_F1

        print(f"\n{'='*80}")
        print("BIAS AMPLIFICATION")
        print(f"{'='*80}")
        print(f"λM (model leakage):     {lambda_M:.4f}")
        print(f"λD(F1) (dataset @ F1):  {lambda_D_F1:.4f}")
        print(f"Δ (amplification):      {delta:+.4f}")

        if delta > 0.05:
            print(f"  → SIGNIFICANT BIAS AMPLIFICATION")
        elif delta > 0:
            print(f"  → Moderate bias amplification")
        else:
            print(f"  → No bias amplification")

        return delta


def comprehensive_bias_evaluation(generated_captions: List[str],
                                  ground_truth_captions: List[str],
                                  race_labels: List[int],
                                  vocab,
                                  dataset_stats: Dict,
                                  image_ids: List = None) -> Dict:
    """
    Comprehensive bias evaluation matching the paper's Table 1.

    Args:
        generated_captions: Model-generated captions
        ground_truth_captions: Ground truth captions
        race_labels: Race labels (0=Light, 1=Dark)
        vocab: Vocabulary object
        dataset_stats: Dataset statistics (light_count, dark_count)
        image_ids: Optional image IDs to prevent data leakage

    Returns:
        Dictionary with all metrics matching paper's table
    """
    print(f"\n{'#'*80}")
    print("COMPREHENSIVE BIAS EVALUATION (PAPER METHODOLOGY)")
    print(f"{'#'*80}\n")

    metrics = LeakageMetrics()
    results = {}

    # Dataset statistics
    light_count = dataset_stats['light_count']
    dark_count = dataset_stats['dark_count']
    total = light_count + dark_count

    results['statistics'] = {
        '#light': light_count,
        '#dark': dark_count,
        'total': total,
        'light_ratio': light_count / total,
        'dark_ratio': dark_count / total
    }

    print(f"Dataset Statistics:")
    print(f"  Light samples: {light_count} ({light_count/total*100:.1f}%)")
    print(f"  Dark samples:  {dark_count} ({dark_count/total*100:.1f}%)")

    # 1. Dataset Leakage (λD)
    dataset_leak = metrics.compute_dataset_leakage(
        ground_truth_captions, race_labels, image_ids=image_ids
    )
    results['lambda_D'] = dataset_leak['lambda_D']

    # 2. Model Leakage (λM)
    model_leak = metrics.compute_model_leakage(
        generated_captions, race_labels,
        vectorizer=dataset_leak['vectorizer'], image_ids=image_ids
    )
    results['lambda_M'] = model_leak['lambda_M']

    # 3. Compute model F1 score
    from evaluation.caption_metrics import evaluate_captions
    references = [[gt] for gt in ground_truth_captions]
    caption_quality = evaluate_captions(generated_captions, references)
    model_f1 = caption_quality['F1']
    results['F1'] = model_f1
    results['caption_quality'] = caption_quality

    print(f"\nModel Caption Quality:")
    print(f"  BLEU-4: {caption_quality['BLEU-4']:.4f}")
    print(f"  F1:     {caption_quality['F1']:.4f}")

    # 4. Dataset Leakage at F1 (λD(F1))
    dataset_leak_f1 = metrics.compute_dataset_leakage_at_f1(
        ground_truth_captions, references, race_labels,
        model_f1, vocab, vectorizer=dataset_leak['vectorizer'], image_ids=image_ids
    )
    results['lambda_D_F1'] = dataset_leak_f1['lambda_D_F1']

    # 5. Bias Amplification (Δ)
    bias_amp = metrics.compute_bias_amplification(
        results['lambda_M'], results['lambda_D_F1']
    )
    results['bias_amplification'] = bias_amp

    return results


def format_results_table(results: Dict, split_name: str = "baseline") -> str:
    """
    Format results to match paper's Table 1 format.

    Args:
        results: Results dictionary
        split_name: Name of the split/model

    Returns:
        Formatted table string
    """
    stats = results['statistics']

    table = []
    table.append("=" * 100)
    table.append("BIAS EVALUATION RESULTS (Paper Table 1 Format)")
    table.append("=" * 100)
    table.append("")
    table.append(f"{'Split':<15} {'Statistics':<30} {'Leakage':<30} {'Performance':<20}")
    table.append(f"{'':<15} {'#light':<10} {'#dark':<10} {'λD':<10} {'λM(F1)':<10} {'λD(F1)':<10} {'Δ':<10} {'F1':<10}")
    table.append("-" * 100)

    row = f"{split_name:<15} "
    row += f"{stats['#light']:<10} "
    row += f"{stats['#dark']:<10} "
    row += f"{results['lambda_D']:.4f}    "
    row += f"{results['lambda_M']:.4f}    "
    row += f"{results['lambda_D_F1']:.4f}    "
    row += f"{results['bias_amplification']:+.4f}    "
    row += f"{results['F1']:.4f}"

    table.append(row)
    table.append("=" * 100)
    table.append("")
    table.append("INTERPRETATION:")
    table.append(f"  λD      = Dataset Leakage (how predictable is race from ground truth captions)")
    table.append(f"  λM(F1)  = Model Leakage (how predictable is race from generated captions)")
    table.append(f"  λD(F1)  = Dataset Leakage at model's F1 (expected leakage by chance)")
    table.append(f"  Δ       = Bias Amplification (λM - λD(F1))")
    table.append("")

    if results['bias_amplification'] > 0.05:
        table.append("⚠️  VERDICT: Model AMPLIFIES racial bias beyond what exists in the dataset")
    elif results['bias_amplification'] > 0:
        table.append("⚠️  VERDICT: Model shows moderate bias amplification")
    else:
        table.append("✅ VERDICT: Model does not amplify bias")

    table.append("=" * 100)

    return '\n'.join(table)


# Backward compatibility - keep old functions but deprecated
def print_bias_summary(title: str = "Bias Evaluation Summary"):
    """Legacy function for backward compatibility."""
    print(f"\n{title}")
    print("Note: This is using the old (incorrect) methodology.")
    print("Please use comprehensive_bias_evaluation() for paper-correct metrics.")


if __name__ == '__main__':
    print("Testing LeakageMetrics...")

    # Create dummy data
    captions_gt = [
        "a man playing tennis",
        "a woman cooking food",
        "a person riding bike",
        "a man with dog"
    ] * 25

    captions_gen = [
        "man with racket",
        "woman in kitchen",
        "person on bicycle",
        "man and pet"
    ] * 25

    race_labels = [0, 1, 0, 1] * 25  # 0=light, 1=dark

    # Mock vocab
    class MockVocab:
        def __init__(self):
            self.idx2word = {i: word for i, word in enumerate(
                "a an the man woman person dog cat playing cooking riding".split()
            )}

        def __len__(self):
            return len(self.idx2word)

    vocab = MockVocab()

    dataset_stats = {
        'light_count': 50,
        'dark_count': 50
    }

    results = comprehensive_bias_evaluation(
        captions_gen, captions_gt, race_labels, vocab, dataset_stats
    )

    print("\n" + format_results_table(results))
