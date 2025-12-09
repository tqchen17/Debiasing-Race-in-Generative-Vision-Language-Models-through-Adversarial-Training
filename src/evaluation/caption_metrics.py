"""
Caption quality evaluation metrics: BLEU and ROUGE.
"""

import numpy as np
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CaptionMetrics:
    """
    Evaluate caption quality using BLEU and ROUGE metrics.
    """

    def __init__(self):
        """Initialize metrics."""
        self.smooth = SmoothingFunction().method4
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_bleu(self, references: List[List[str]], hypothesis: List[str],
                     weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
        """
        Compute BLEU score for a single caption.

        Args:
            references: List of reference captions (each is list of words)
            hypothesis: Generated caption (list of words)
            weights: Weights for n-grams (default: BLEU-4)

        Returns:
            BLEU score
        """
        return sentence_bleu(
            references,
            hypothesis,
            weights=weights,
            smoothing_function=self.smooth
        )

    def compute_corpus_bleu(self, all_references: List[List[List[str]]],
                           all_hypotheses: List[List[str]],
                           max_n: int = 4) -> Dict[str, float]:
        """
        Compute corpus-level BLEU scores.

        Args:
            all_references: List of reference lists for each sample
            all_hypotheses: List of hypothesis captions
            max_n: Maximum n-gram order

        Returns:
            Dictionary with BLEU-1, 2, 3, 4 scores
        """
        bleu_scores = {}

        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            score = corpus_bleu(
                all_references,
                all_hypotheses,
                weights=weights,
                smoothing_function=self.smooth
            )
            bleu_scores[f'BLEU-{n}'] = score

        return bleu_scores

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            reference: Reference caption (string)
            hypothesis: Generated caption (string)

        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge.score(reference, hypothesis)

        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }

    def compute_all_metrics(self, references_list: List[List[str]],
                          hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute all caption quality metrics.

        Args:
            references_list: List of reference captions for each sample
                            (each sample can have multiple references)
            hypotheses: List of generated captions

        Returns:
            Dictionary with all metrics
        """
        # Convert to format for BLEU
        # references_list: [[ref1_words, ref2_words, ...], ...]
        # hypotheses: [hyp1_words, hyp2_words, ...]

        # For corpus BLEU
        bleu_scores = self.compute_corpus_bleu(references_list, hypotheses)

        # For ROUGE, average across all samples
        rouge_scores = []
        for refs, hyp in zip(references_list, hypotheses):
            # Use first reference for ROUGE
            ref_str = ' '.join(refs[0])
            hyp_str = ' '.join(hyp)
            rouge_scores.append(self.compute_rouge(ref_str, hyp_str))

        # Average ROUGE scores
        avg_rouge = {
            'ROUGE-1': np.mean([s['ROUGE-1'] for s in rouge_scores]),
            'ROUGE-2': np.mean([s['ROUGE-2'] for s in rouge_scores]),
            'ROUGE-L': np.mean([s['ROUGE-L'] for s in rouge_scores])
        }

        # Combine all metrics
        all_metrics = {**bleu_scores, **avg_rouge}

        # Add F1 as alias for ROUGE-L (standard practice in caption evaluation)
        all_metrics['F1'] = avg_rouge['ROUGE-L']

        return all_metrics


def evaluate_captions(generated_captions: List[str],
                     reference_captions: List[List[str]],
                     tokenize: bool = True) -> Dict[str, float]:
    """
    Evaluate generated captions against references.

    Args:
        generated_captions: List of generated caption strings
        reference_captions: List of reference caption lists (multiple refs per sample)
        tokenize: Whether to tokenize captions (if False, assumes already tokenized)

    Returns:
        Dictionary with all metrics
    """
    metrics = CaptionMetrics()

    if tokenize:
        # Tokenize captions
        generated_tokens = [cap.lower().split() for cap in generated_captions]
        reference_tokens = [
            [ref.lower().split() for ref in refs]
            for refs in reference_captions
        ]
    else:
        generated_tokens = generated_captions
        reference_tokens = reference_captions

    # Compute metrics
    results = metrics.compute_all_metrics(reference_tokens, generated_tokens)

    return results


def print_metrics(metrics: Dict[str, float], title: str = "Caption Quality Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

    # BLEU scores
    print("\nBLEU Scores:")
    for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # ROUGE scores
    print("\nROUGE Scores:")
    for key in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    print("="*60)


if __name__ == '__main__':
    # Test caption metrics
    print("Testing Caption Quality Metrics...\n")

    # Example captions
    references = [
        ["a man riding a bike on the street", "a person on a bicycle"],
        ["a dog playing with a ball", "a dog chasing a ball in the yard"],
        ["a woman holding an umbrella", "a lady with an umbrella standing"]
    ]

    hypotheses = [
        "a man on a bike",
        "a dog playing with a ball",
        "a woman with an umbrella"
    ]

    # Compute metrics
    metrics = evaluate_captions(hypotheses, [[r] if isinstance(r, str) else r for r in references])

    # Print results
    print_metrics(metrics, "Test Caption Metrics")

    print("\n✅ Caption metrics test passed!")
