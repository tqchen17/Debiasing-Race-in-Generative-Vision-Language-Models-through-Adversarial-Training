"""
Vocabulary builder for image captions.
Builds vocabulary from training captions and provides tokenization utilities.
"""

import json
import pickle
from collections import Counter
from typing import List, Dict
import pandas as pd


class Vocabulary:
    """Vocabulary class for text tokenization."""

    # Special tokens
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    UNK_TOKEN = '<UNK>'

    def __init__(self, min_freq: int = 5):
        """
        Initialize vocabulary.

        Args:
            min_freq: Minimum frequency for a word to be included in vocabulary
        """
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Initialize with special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.START_TOKEN,
            self.END_TOKEN,
            self.UNK_TOKEN
        ]
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token

    def build_vocab(self, captions: List[str]):
        """
        Build vocabulary from list of captions.

        Args:
            captions: List of caption strings
        """
        # Count word frequencies
        for caption in captions:
            words = self._tokenize(caption)
            self.word_freq.update(words)

        # Add words that meet minimum frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

        print(f"Vocabulary built with {len(self.word2idx)} words")
        print(f"Min frequency threshold: {self.min_freq}")
        print(f"Total unique words seen: {len(self.word_freq)}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of lowercase words
        """
        # Simple tokenization: lowercase and split
        # Remove punctuation and convert to lowercase
        text = text.lower()
        # Remove common punctuation
        for char in ['.', ',', '!', '?', ';', ':']:
            text = text.replace(char, '')
        return text.split()

    def encode(self, caption: str, max_length: int = None,
               add_start_end: bool = True) -> List[int]:
        """
        Encode caption to list of indices.

        Args:
            caption: Caption string to encode
            max_length: Maximum sequence length (if None, no truncation)
            add_start_end: Whether to add START and END tokens

        Returns:
            List of word indices
        """
        words = self._tokenize(caption)

        # Convert words to indices
        indices = []
        if add_start_end:
            indices.append(self.word2idx[self.START_TOKEN])

        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])

        if add_start_end:
            indices.append(self.word2idx[self.END_TOKEN])

        # Truncate if needed
        if max_length is not None and len(indices) > max_length:
            indices = indices[:max_length]
            # Ensure END token is present if we're adding start/end
            if add_start_end:
                indices[-1] = self.word2idx[self.END_TOKEN]

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode list of indices to caption string.

        Args:
            indices: List of word indices
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded caption string
        """
        words = []
        special_tokens = {self.PAD_TOKEN, self.START_TOKEN,
                         self.END_TOKEN, self.UNK_TOKEN}

        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if skip_special_tokens and word in special_tokens:
                    continue
                if word == self.END_TOKEN:
                    break
                words.append(word)

        return ' '.join(words)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        """Return padding token index."""
        return self.word2idx[self.PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        """Return start token index."""
        return self.word2idx[self.START_TOKEN]

    @property
    def end_idx(self) -> int:
        """Return end token index."""
        return self.word2idx[self.END_TOKEN]

    @property
    def unk_idx(self) -> int:
        """Return unknown token index."""
        return self.word2idx[self.UNK_TOKEN]

    def save(self, filepath: str):
        """
        Save vocabulary to file.

        Args:
            filepath: Path to save vocabulary (pickle format)
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'min_freq': self.min_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        Load vocabulary from file.

        Args:
            filepath: Path to vocabulary file

        Returns:
            Loaded Vocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(min_freq=vocab_data['min_freq'])
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = Counter(vocab_data['word_freq'])

        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(vocab)}")
        return vocab


def build_vocab_from_csv(csv_path: str, min_freq: int = 5,
                         save_path: str = None) -> Vocabulary:
    """
    Build vocabulary from CSV file containing captions.

    Args:
        csv_path: Path to CSV file with 'caption_text' column
        min_freq: Minimum frequency for word inclusion
        save_path: Optional path to save vocabulary

    Returns:
        Built Vocabulary object
    """
    print(f"Building vocabulary from {csv_path}")

    # Load captions
    df = pd.read_csv(csv_path)
    captions = df['caption_text'].tolist()

    # Build vocabulary
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(captions)

    # Save if path provided
    if save_path:
        vocab.save(save_path)

    return vocab


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Build vocabulary from training captions')
    parser.add_argument('--train_csv', type=str, default='MASTER_TRAIN.csv',
                       help='Path to training CSV file')
    parser.add_argument('--min_freq', type=int, default=5,
                       help='Minimum word frequency')
    parser.add_argument('--save_path', type=str, default='vocab.pkl',
                       help='Path to save vocabulary')

    args = parser.parse_args()

    # Build and save vocabulary
    vocab = build_vocab_from_csv(
        csv_path=args.train_csv,
        min_freq=args.min_freq,
        save_path=args.save_path
    )

    # Print some statistics
    print(f"\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"PAD index: {vocab.pad_idx}")
    print(f"START index: {vocab.start_idx}")
    print(f"END index: {vocab.end_idx}")
    print(f"UNK index: {vocab.unk_idx}")

    # Test encoding/decoding
    test_caption = "A man riding a bike on the street."
    encoded = vocab.encode(test_caption, max_length=20)
    decoded = vocab.decode(encoded)
    print(f"\nTest encoding/decoding:")
    print(f"Original: {test_caption}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
