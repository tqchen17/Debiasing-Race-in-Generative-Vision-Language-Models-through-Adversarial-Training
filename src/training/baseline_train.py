"""
Training script for baseline image captioning model (no debiasing).
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_datasets
from models.baseline_model import create_baseline_model, compute_loss
from utils.vocab import Vocabulary
import utils.config as config


class BaselineTrainer:
    """Trainer for baseline image captioning model."""

    def __init__(self, vocab, model, train_dataset, val_dataset,
                 learning_rate=config.LR_DECODER, checkpoint_dir=None,
                 log_dir=None):
        """
        Initialize trainer.

        Args:
            vocab: Vocabulary object
            model: Baseline model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.vocab = vocab
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate)

        # Loss function
        self.loss_object = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.val_loss = keras.metrics.Mean(name='val_loss')

        # Checkpoints
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=5
        )

        # TensorBoard
        self.log_dir = log_dir or os.path.join(config.LOG_DIR, 'baseline')
        os.makedirs(self.log_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(self.log_dir, current_time, 'train')
        self.val_log_dir = os.path.join(self.log_dir, current_time, 'val')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        print(f"Trainer initialized:")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  TensorBoard log dir: {self.log_dir}")
        print(f"  Learning rate: {learning_rate}")

    @tf.function
    def train_step(self, images, captions):
        """
        Single training step.

        Args:
            images: Batch of images [B, 224, 224, 3]
            captions: Batch of captions [B, max_length]

        Returns:
            loss: Batch loss
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions, features, hidden_states = self.model(
                images, captions, training=True
            )

            # Compute loss
            # Shift captions: input = [<START>, w1, w2, ...], target = [w1, w2, ..., <END>]
            target = captions[:, 1:]  # Remove <START>
            pred = predictions[:, :-1, :]  # Remove last prediction

            loss = compute_loss(target, pred, self.loss_object)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, config.GRAD_CLIP)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss

    @tf.function
    def val_step(self, images, captions):
        """
        Single validation step.

        Args:
            images: Batch of images
            captions: Batch of captions

        Returns:
            loss: Batch loss
        """
        # Forward pass
        predictions, features, hidden_states = self.model(
            images, captions, training=False
        )

        # Compute loss
        target = captions[:, 1:]
        pred = predictions[:, :-1, :]
        loss = compute_loss(target, pred, self.loss_object)

        return loss

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.train_loss.reset_state()

        start_time = time.time()
        batch_count = 0

        for batch in self.train_dataset:
            images = batch['image']
            captions = batch['caption']

            # Train step
            loss = self.train_step(images, captions)
            self.train_loss.update_state(loss)

            batch_count += 1

            # Log every N steps
            if batch_count % config.LOG_EVERY_N_STEPS == 0:
                print(f"  Batch {batch_count}: Loss = {self.train_loss.result():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = self.train_loss.result()

        print(f"\nEpoch {epoch + 1} Training:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

        # TensorBoard logging
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('epoch_time', epoch_time, step=epoch)

        return avg_loss.numpy()

    def validate(self, epoch):
        """
        Run validation.

        Args:
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.val_loss.reset_state()

        for batch in self.val_dataset:
            images = batch['image']
            captions = batch['caption']

            loss = self.val_step(images, captions)
            self.val_loss.update_state(loss)

        avg_loss = self.val_loss.result()

        print(f"Epoch {epoch + 1} Validation:")
        print(f"  Loss: {avg_loss:.4f}")

        # TensorBoard logging
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)

        return avg_loss.numpy()

    def generate_sample_captions(self, epoch, num_samples=5):
        """
        Generate sample captions for visualization.

        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
        """
        print(f"\nSample Captions (Epoch {epoch + 1}):")
        print("-" * 80)

        count = 0
        for batch in self.val_dataset.take(1):
            images = batch['image']
            captions = batch['caption']
            race_labels = batch['race_label']

            for i in range(min(num_samples, len(images))):
                # Generate caption
                single_image = tf.expand_dims(images[i], 0)
                generated = self.model.generate_caption(
                    single_image,
                    start_token=self.vocab.start_idx,
                    end_token=self.vocab.end_idx,
                    max_length=20
                )

                # Decode captions
                generated_text = self.vocab.decode(generated)
                reference_text = self.vocab.decode(captions[i].numpy())
                race = "Light" if race_labels[i] == 0 else "Dark"

                print(f"\n{count + 1}. Race: {race}")
                print(f"   Generated: {generated_text}")
                print(f"   Reference: {reference_text}")

                count += 1
                if count >= num_samples:
                    break

        print("-" * 80)

    def train(self, num_epochs=config.NUM_EPOCHS, save_every=config.SAVE_EVERY_N_EPOCHS):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print("\n" + "=" * 80)
        print("BASELINE MODEL TRAINING")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LR_DECODER}")
        print(f"Vocab size: {len(self.vocab)}")
        print("=" * 80 + "\n")

        # Restore checkpoint if exists
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}\n")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Generate sample captions
            if (epoch + 1) % 5 == 0:
                self.generate_sample_captions(epoch)

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                save_path = self.checkpoint_manager.save()
                print(f"\nSaved checkpoint: {save_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, 'best_model.weights.h5')
                self.model.save_weights(best_path)
                print(f"Saved best model (val_loss={val_loss:.4f})")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir}")

        return best_val_loss


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train baseline image captioning model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced sampling')
    parser.add_argument('--finetune_encoder', action='store_true',
                       help='Finetune encoder')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='TensorBoard log directory')

    args = parser.parse_args()

    # Update config
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR_DECODER = args.lr

    print("Loading datasets...")
    train_dataset, val_dataset, vocab = create_datasets(
        batch_size=args.batch_size,
        balanced=args.balanced
    )

    print("\nCreating model...")
    model = create_baseline_model(
        vocab_size=len(vocab),
        finetune_encoder=args.finetune_encoder
    )

    print("\nInitializing trainer...")
    trainer = BaselineTrainer(
        vocab=vocab,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.epochs)

    print("\nTraining finished! Run evaluation script for bias analysis.")


if __name__ == '__main__':
    main()
