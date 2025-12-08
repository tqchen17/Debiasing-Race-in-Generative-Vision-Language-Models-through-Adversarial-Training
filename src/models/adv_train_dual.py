"""
Training script for dual-location adversarial debiasing model.
Implements training loop with gradient reversal at encoder and decoder.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adv_model_dual import create_dual_adversarial_model, compute_dual_loss
from data.dataset import create_dataset
from utils.vocab import load_vocab
import utils.config as config


class DualAdversarialTrainer:
    """
    Trainer for dual-location adversarial debiasing.
    """
    
    def __init__(self, model, train_dataset, val_dataset,
                 lr_encoder=1e-4, lr_decoder=1e-3, lr_adversary=1e-3,
                 lambda_v=0.5, lambda_d=0.5,
                 checkpoint_dir='checkpoints/dual_adversarial',
                 log_dir='logs/dual_adversarial',
                 use_temporal_adversary=False):
        """
        Initialize trainer.
        
        Args:
            model: DualAdversarialModel instance
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            lr_encoder: Learning rate for encoder (if finetuning)
            lr_decoder: Learning rate for decoder
            lr_adversary: Learning rate for adversaries
            lambda_v: Weight for encoder adversary
            lambda_d: Weight for decoder adversary
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            use_temporal_adversary: Whether to use temporal averaging
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lambda_v = lambda_v
        self.lambda_d = lambda_d
        self.use_temporal_adversary = use_temporal_adversary
        
        # Create optimizers
        self.encoder_optimizer = keras.optimizers.Adam(learning_rate=lr_encoder)
        self.decoder_optimizer = keras.optimizers.Adam(learning_rate=lr_decoder)
        self.adversary_optimizer = keras.optimizers.Adam(learning_rate=lr_adversary)
        
        # Loss function
        self.caption_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
        
        # Metrics
        self.train_caption_loss = keras.metrics.Mean(name='train_caption_loss')
        self.train_adv_loss_v = keras.metrics.Mean(name='train_adv_loss_v')
        self.train_adv_loss_d = keras.metrics.Mean(name='train_adv_loss_d')
        self.train_adv_acc_v = keras.metrics.SparseCategoricalAccuracy(name='train_adv_acc_v')
        self.train_adv_acc_d = keras.metrics.SparseCategoricalAccuracy(name='train_adv_acc_d')
        
        self.val_caption_loss = keras.metrics.Mean(name='val_caption_loss')
        self.val_adv_loss_v = keras.metrics.Mean(name='val_adv_loss_v')
        self.val_adv_loss_d = keras.metrics.Mean(name='val_adv_loss_d')
        self.val_adv_acc_v = keras.metrics.SparseCategoricalAccuracy(name='val_adv_acc_v')
        self.val_adv_acc_d = keras.metrics.SparseCategoricalAccuracy(name='val_adv_acc_d')
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint = tf.train.Checkpoint(
            model=model,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            adversary_optimizer=self.adversary_optimizer
        )
        
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=5
        )
        
        # TensorBoard
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        print(f"\nDualAdversarialTrainer initialized:")
        print(f"  Encoder LR: {lr_encoder}")
        print(f"  Decoder LR: {lr_decoder}")
        print(f"  Adversary LR: {lr_adversary}")
        print(f"  Lambda encoder (λ_v): {lambda_v}")
        print(f"  Lambda decoder (λ_d): {lambda_d}")
        print(f"  Use temporal adversary: {use_temporal_adversary}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Log dir: {log_dir}")
    
    @tf.function
    def train_step(self, images, captions, race_labels):
        """
        Single training step with dual adversarial training.
        
        Args:
            images: Batch of images [batch_size, 224, 224, 3]
            captions: Batch of captions [batch_size, max_length]
            race_labels: Batch of race labels [batch_size]
        
        Returns:
            Tuple of losses and accuracies
        """
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            if self.use_temporal_adversary:
                (predictions, encoder_features, decoder_hiddens,
                 race_pred_encoder, race_pred_decoder) = \
                    self.model.call_with_temporal_adversary(
                        images, captions, training=True
                    )
            else:
                (predictions, encoder_features, decoder_hiddens,
                 race_pred_encoder, race_pred_decoder) = \
                    self.model(images, captions, training=True)
            
            # Shift captions for next word prediction
            real = captions[:, 1:]
            pred = predictions[:, :-1, :]
            
            # Compute losses
            total_loss, caption_loss, adv_loss_v, adv_loss_d = compute_dual_loss(
                real, pred, race_labels,
                race_pred_encoder, race_pred_decoder,
                self.caption_loss_fn,
                self.lambda_v, self.lambda_d
            )
        
        # Compute gradients for main model
        main_vars = (self.model.encoder.trainable_variables + 
                    self.model.decoder.trainable_variables)
        main_gradients = tape.gradient(total_loss, main_vars)
        
        # Clip gradients
        main_gradients, _ = tf.clip_by_global_norm(main_gradients, config.GRAD_CLIP)
        
        # Apply gradients to encoder
        if self.model.finetune_encoder:
            encoder_grads = main_gradients[:len(self.model.encoder.trainable_variables)]
            self.encoder_optimizer.apply_gradients(
                zip(encoder_grads, self.model.encoder.trainable_variables)
            )
        
        # Apply gradients to decoder
        decoder_start = len(self.model.encoder.trainable_variables)
        decoder_grads = main_gradients[decoder_start:]
        self.decoder_optimizer.apply_gradients(
            zip(decoder_grads, self.model.decoder.trainable_variables)
        )
        
        # Compute adversary accuracies
        adv_acc_v = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(race_pred_encoder, axis=-1, output_type=tf.int32),
                    race_labels
                ),
                tf.float32
            )
        )
        
        adv_acc_d = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(race_pred_decoder, axis=-1, output_type=tf.int32),
                    race_labels
                ),
                tf.float32
            )
        )
        
        del tape
        
        return caption_loss, adv_loss_v, adv_loss_d, adv_acc_v, adv_acc_d
    
    @tf.function
    def val_step(self, images, captions, race_labels):
        """
        Single validation step.
        
        Args:
            images: Batch of images
            captions: Batch of captions
            race_labels: Batch of race labels
        
        Returns:
            Tuple of losses and accuracies
        """
        # Forward pass
        if self.use_temporal_adversary:
            (predictions, encoder_features, decoder_hiddens,
             race_pred_encoder, race_pred_decoder) = \
                self.model.call_with_temporal_adversary(
                    images, captions, training=False
                )
        else:
            (predictions, encoder_features, decoder_hiddens,
             race_pred_encoder, race_pred_decoder) = \
                self.model(images, captions, training=False)
        
        # Shift captions
        real = captions[:, 1:]
        pred = predictions[:, :-1, :]
        
        # Compute losses
        _, caption_loss, adv_loss_v, adv_loss_d = compute_dual_loss(
            real, pred, race_labels,
            race_pred_encoder, race_pred_decoder,
            self.caption_loss_fn,
            self.lambda_v, self.lambda_d
        )
        
        # Compute accuracies
        adv_acc_v = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(race_pred_encoder, axis=-1, output_type=tf.int32),
                    race_labels
                ),
                tf.float32
            )
        )
        
        adv_acc_d = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(race_pred_decoder, axis=-1, output_type=tf.int32),
                    race_labels
                ),
                tf.float32
            )
        )
        
        return caption_loss, adv_loss_v, adv_loss_d, adv_acc_v, adv_acc_d
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average metrics
        """
        # Reset metrics
        self.train_caption_loss.reset_state()
        self.train_adv_loss_v.reset_state()
        self.train_adv_loss_d.reset_state()
        
        # Training loop
        num_batches = 0
        start_time = time.time()
        
        for batch in self.train_dataset:
            images = batch['image']
            captions = batch['caption']
            race_labels = batch['race_label']
            
            # Train step
            cap_loss, adv_loss_v, adv_loss_d, adv_acc_v, adv_acc_d = \
                self.train_step(images, captions, race_labels)
            
            # Update metrics
            self.train_caption_loss.update_state(cap_loss)
            self.train_adv_loss_v.update_state(adv_loss_v)
            self.train_adv_loss_d.update_state(adv_loss_d)
            
            num_batches += 1
            
            # Print progress every 100 batches
            if num_batches % 100 == 0:
                print(f"  Batch {num_batches}: "
                      f"cap_loss={cap_loss:.4f}, "
                      f"adv_v={adv_loss_v:.4f}(acc={adv_acc_v:.3f}), "
                      f"adv_d={adv_loss_d:.4f}(acc={adv_acc_d:.3f})")
        
        epoch_time = time.time() - start_time
        
        # Get average metrics
        metrics = {
            'caption_loss': self.train_caption_loss.result().numpy(),
            'adv_loss_v': self.train_adv_loss_v.result().numpy(),
            'adv_loss_d': self.train_adv_loss_d.result().numpy(),
            'time': epoch_time
        }
        
        return metrics
    
    def validate_epoch(self):
        """
        Validate on validation set.
        
        Returns:
            Dictionary of average metrics
        """
        # Reset metrics
        self.val_caption_loss.reset_state()
        self.val_adv_loss_v.reset_state()
        self.val_adv_loss_d.reset_state()
        
        # Validation loop
        for batch in self.val_dataset:
            images = batch['image']
            captions = batch['caption']
            race_labels = batch['race_label']
            
            # Validation step
            cap_loss, adv_loss_v, adv_loss_d, adv_acc_v, adv_acc_d = \
                self.val_step(images, captions, race_labels)
            
            # Update metrics
            self.val_caption_loss.update_state(cap_loss)
            self.val_adv_loss_v.update_state(adv_loss_v)
            self.val_adv_loss_d.update_state(adv_loss_d)
        
        # Get average metrics
        metrics = {
            'caption_loss': self.val_caption_loss.result().numpy(),
            'adv_loss_v': self.val_adv_loss_v.result().numpy(),
            'adv_loss_d': self.val_adv_loss_d.result().numpy()
        }
        
        return metrics
    
    def train(self, num_epochs):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train - Caption Loss: {train_metrics['caption_loss']:.4f}")
            print(f"  Train - Adv Loss V: {train_metrics['adv_loss_v']:.4f}")
            print(f"  Train - Adv Loss D: {train_metrics['adv_loss_d']:.4f}")
            print(f"  Val   - Caption Loss: {val_metrics['caption_loss']:.4f}")
            print(f"  Val   - Adv Loss V: {val_metrics['adv_loss_v']:.4f}")
            print(f"  Val   - Adv Loss D: {val_metrics['adv_loss_d']:.4f}")
            print(f"  Time: {train_metrics['time']:.2f}s")
            
            # Log to TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('train/caption_loss', train_metrics['caption_loss'], step=epoch)
                tf.summary.scalar('train/adv_loss_v', train_metrics['adv_loss_v'], step=epoch)
                tf.summary.scalar('train/adv_loss_d', train_metrics['adv_loss_d'], step=epoch)
                tf.summary.scalar('val/caption_loss', val_metrics['caption_loss'], step=epoch)
                tf.summary.scalar('val/adv_loss_v', val_metrics['adv_loss_v'], step=epoch)
                tf.summary.scalar('val/adv_loss_d', val_metrics['adv_loss_d'], step=epoch)
            
            # Save checkpoint
            if val_metrics['caption_loss'] < best_val_loss:
                best_val_loss = val_metrics['caption_loss']
                save_path = self.checkpoint_manager.save()
                print(f"\n✅ Saved checkpoint: {save_path}")
            
            # Save checkpoint every 5 epochs anyway
            if (epoch + 1) % 5 == 0:
                save_path = self.checkpoint_manager.save()
                print(f"\n💾 Saved periodic checkpoint: {save_path}")
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best validation caption loss: {best_val_loss:.4f}")
        print(f"{'='*70}\n")


def main():
    """Main training function."""
    
    print("="*70)
    print("DUAL ADVERSARIAL DEBIASING TRAINING")
    print("="*70)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    word_to_idx, idx_to_word = load_vocab(config.VOCAB_FILE)
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_dataset(
        csv_file=config.TRAIN_CSV,
        batch_size=config.BATCH_SIZE,
        vocab_file=config.VOCAB_FILE,
        max_length=config.MAX_CAPTION_LENGTH,
        is_training=True
    )
    
    val_dataset = create_dataset(
        csv_file=config.VAL_CSV,
        batch_size=config.BATCH_SIZE,
        vocab_file=config.VOCAB_FILE,
        max_length=config.MAX_CAPTION_LENGTH,
        is_training=False
    )
    
    print("Datasets created")
    
    # Create model
    print("\nCreating dual adversarial model...")
    model = create_dual_adversarial_model(
        vocab_size=vocab_size,
        lambda_v=config.LAMBDA_V,
        lambda_d=config.LAMBDA_D,
        finetune_encoder=config.FINETUNE_ENCODER
    )
    print("Model created")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = DualAdversarialTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr_encoder=config.LR_ENCODER,
        lr_decoder=config.LR_DECODER,
        lr_adversary=config.LR_ADVERSARY,
        lambda_v=config.LAMBDA_V,
        lambda_d=config.LAMBDA_D,
        checkpoint_dir=config.DUAL_CHECKPOINT_DIR,
        log_dir=config.DUAL_LOG_DIR,
        use_temporal_adversary=config.USE_TEMPORAL_ADVERSARY
    )
    print("Trainer initialized")
    
    # Train
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()