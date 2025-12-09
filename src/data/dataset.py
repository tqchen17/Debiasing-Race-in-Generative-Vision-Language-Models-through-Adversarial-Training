"""
TensorFlow data pipeline for image captioning with race labels.
Implements efficient data loading, preprocessing, and balanced batch sampling.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vocab import Vocabulary
import utils.config as config


class ImageCaptionDataset:
    """Dataset class for image captioning with race labels."""

    def __init__(self, csv_path: str, vocab: Vocabulary,
                 max_caption_length: int = config.MAX_CAPTION_LENGTH,
                 image_size: int = config.IMAGE_SIZE,
                 augment: bool = True):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file
            vocab: Vocabulary object
            max_caption_length: Maximum caption length
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        self.csv_path = csv_path
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.image_size = image_size
        self.augment = augment

        # Load dataframe
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Print race distribution
        race_dist = self.df['race_label'].value_counts()
        print(f"Race distribution: Light={race_dist.get(0, 0)}, Dark={race_dist.get(1, 0)}")

    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor [H, W, 3]
        """
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Resize
        if self.augment:
            # Resize to slightly larger size for random crop
            img = tf.image.resize(img, [self.image_size + 32, self.image_size + 32])
            # Random crop
            img = tf.image.random_crop(img, [self.image_size, self.image_size, 3])
            # Random horizontal flip
            img = tf.image.random_flip_left_right(img)
        else:
            # Just resize for validation
            img = tf.image.resize(img, [self.image_size, self.image_size])

        # Convert to float32
        img = tf.cast(img, tf.float32)

        # Apply ImageNet preprocessing for ResNet50
        # This converts to BGR and subtracts ImageNet mean
        img = tf.keras.applications.resnet50.preprocess_input(img)

        return img

    def encode_caption(self, caption: str) -> Tuple[tf.Tensor, int]:
        """
        Encode caption to sequence of indices.

        Args:
            caption: Caption string

        Returns:
            Tuple of (encoded caption, original length)
        """
        # Encode caption using vocabulary
        encoded = self.vocab.encode(caption, max_length=self.max_caption_length,
                                   add_start_end=True)

        # Pad to max length
        original_length = len(encoded)
        if len(encoded) < self.max_caption_length:
            encoded = encoded + [self.vocab.pad_idx] * (self.max_caption_length - len(encoded))
        else:
            encoded = encoded[:self.max_caption_length]
            original_length = self.max_caption_length

        return np.array(encoded, dtype=np.int32), original_length

    def create_tf_dataset(self, batch_size: int = config.BATCH_SIZE,
                         shuffle: bool = True,
                         buffer_size: int = 1000) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            buffer_size: Shuffle buffer size

        Returns:
            TensorFlow dataset
        """
        # Create dataset from dataframe
        image_paths = self.df['image_path'].values
        captions = self.df['caption_text'].values
        race_labels = self.df['race_label'].values
        image_ids = self.df['id'].values if 'id' in self.df.columns else np.arange(len(self.df))

        def generator():
            """Generator function for tf.data.Dataset."""
            for img_path, caption, race_label, img_id in zip(image_paths, captions, race_labels, image_ids):
                # Encode caption
                encoded_caption, caption_length = self.encode_caption(caption)

                yield (
                    img_path,
                    encoded_caption,
                    caption_length,
                    race_label,
                    img_id
                )

        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),  # image_path
                tf.TensorSpec(shape=(self.max_caption_length,), dtype=tf.int32),  # caption
                tf.TensorSpec(shape=(), dtype=tf.int32),  # caption_length
                tf.TensorSpec(shape=(), dtype=tf.int32),  # race_label
                tf.TensorSpec(shape=(), dtype=tf.int32),  # image_id
            )
        )

        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Map preprocessing function
        def load_and_preprocess(image_path, caption, caption_length, race_label, image_id):
            image = self.preprocess_image(image_path)
            return {
                'image': image,
                'caption': caption,
                'caption_length': caption_length,
                'race_label': race_label,
                'image_id': image_id
            }

        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch
        dataset = dataset.batch(batch_size)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_balanced_dataset(self, batch_size: int = config.BATCH_SIZE,
                               buffer_size: int = 1000) -> tf.data.Dataset:
        """
        Create balanced TensorFlow dataset with equal samples from each race.

        Args:
            batch_size: Batch size (should be even for 50/50 split)
            buffer_size: Shuffle buffer size

        Returns:
            Balanced TensorFlow dataset
        """
        # Split data by race
        df_light = self.df[self.df['race_label'] == 0]
        df_dark = self.df[self.df['race_label'] == 1]

        print(f"Creating balanced dataset:")
        print(f"  Light samples: {len(df_light)}")
        print(f"  Dark samples: {len(df_dark)}")

        # Create separate datasets
        def create_race_dataset(df, race_label):
            """Create dataset for specific race."""
            image_paths = df['image_path'].values
            captions = df['caption_text'].values
            image_ids = df['id'].values if 'id' in df.columns else np.arange(len(df))

            def generator():
                for img_path, caption, img_id in zip(image_paths, captions, image_ids):
                    encoded_caption, caption_length = self.encode_caption(caption)
                    yield (img_path, encoded_caption, caption_length, race_label, img_id)

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string),
                    tf.TensorSpec(shape=(self.max_caption_length,), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                )
            )
            return dataset

        dataset_light = create_race_dataset(df_light, 0)
        dataset_dark = create_race_dataset(df_dark, 1)

        # Shuffle each dataset
        dataset_light = dataset_light.shuffle(buffer_size=buffer_size).repeat()
        dataset_dark = dataset_dark.shuffle(buffer_size=buffer_size).repeat()

        # Calculate samples per batch for each class
        samples_per_class = batch_size // 2

        # Batch each dataset
        dataset_light = dataset_light.batch(samples_per_class)
        dataset_dark = dataset_dark.batch(samples_per_class)

        # Interleave the two datasets
        dataset = tf.data.Dataset.zip((dataset_light, dataset_dark))

        # Combine batches
        def merge_batches(light_batch, dark_batch):
            """Merge light and dark batches."""
            img_paths = tf.concat([light_batch[0], dark_batch[0]], axis=0)
            captions = tf.concat([light_batch[1], dark_batch[1]], axis=0)
            caption_lengths = tf.concat([light_batch[2], dark_batch[2]], axis=0)
            race_labels = tf.concat([light_batch[3], dark_batch[3]], axis=0)
            image_ids = tf.concat([light_batch[4], dark_batch[4]], axis=0)
            return (img_paths, captions, caption_lengths, race_labels, image_ids)

        dataset = dataset.map(merge_batches)

        # Apply preprocessing
        def load_and_preprocess(image_paths, captions, caption_lengths, race_labels, image_ids):
            """Load and preprocess a batch of images."""
            images = tf.map_fn(
                self.preprocess_image,
                image_paths,
                fn_output_signature=tf.TensorSpec(shape=(self.image_size, self.image_size, 3), dtype=tf.float32)
            )
            return {
                'image': images,
                'caption': captions,
                'caption_length': caption_lengths,
                'race_label': race_labels,
                'image_id': image_ids
            }

        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def create_datasets(vocab_path: str = config.VOCAB_PATH,
                   train_csv: str = config.TRAIN_CSV,
                   val_csv: str = config.VAL_CSV,
                   batch_size: int = config.BATCH_SIZE,
                   balanced: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create training and validation datasets.

    Args:
        vocab_path: Path to vocabulary file
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        batch_size: Batch size
        balanced: Whether to use balanced sampling for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}")
    vocab = Vocabulary.load(vocab_path)

    # Create training dataset
    print(f"\nCreating training dataset from {train_csv}")
    train_dataset_obj = ImageCaptionDataset(
        csv_path=train_csv,
        vocab=vocab,
        augment=True
    )

    if balanced:
        train_dataset = train_dataset_obj.create_balanced_dataset(batch_size=batch_size)
    else:
        train_dataset = train_dataset_obj.create_tf_dataset(
            batch_size=batch_size,
            shuffle=True
        )

    # Create validation dataset
    print(f"\nCreating validation dataset from {val_csv}")
    val_dataset_obj = ImageCaptionDataset(
        csv_path=val_csv,
        vocab=vocab,
        augment=False
    )

    val_dataset = val_dataset_obj.create_tf_dataset(
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, val_dataset, vocab


if __name__ == '__main__':
    # Test data pipeline
    print("Testing data pipeline...\n")

    # Create datasets
    train_ds, val_ds, vocab = create_datasets(balanced=False, batch_size=8)

    # Test training dataset
    print("\n" + "="*60)
    print("Testing Training Dataset")
    print("="*60)

    for batch in train_ds.take(1):
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Caption shape: {batch['caption'].shape}")
        print(f"Caption length shape: {batch['caption_length'].shape}")
        print(f"Race label shape: {batch['race_label'].shape}")

        print(f"\nRace labels in batch: {batch['race_label'].numpy()}")
        print(f"Caption lengths: {batch['caption_length'].numpy()}")

        # Decode first caption
        first_caption = batch['caption'][0].numpy()
        decoded = vocab.decode(first_caption)
        print(f"\nFirst caption (decoded): {decoded}")

    # Test validation dataset
    print("\n" + "="*60)
    print("Testing Validation Dataset")
    print("="*60)

    for batch in val_ds.take(1):
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Race labels: {batch['race_label'].numpy()}")

    print("\n" + "="*60)
    print("Data pipeline test completed successfully!")
    print("="*60)
