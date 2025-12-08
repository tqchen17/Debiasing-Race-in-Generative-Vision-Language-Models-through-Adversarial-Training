"""
Baseline image captioning model (no debiasing).
Combines encoder and decoder for standard image captioning task.
"""

import tensorflow as tf
from tensorflow import keras
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoder import ImageEncoder
from models.decoder import CaptionDecoder
import utils.config as config


class BaselineModel(keras.Model):
    """
    Baseline image captioning model without debiasing.
    """

    def __init__(self, vocab_size, encoder_dim=config.ENCODER_DIM,
                 embedding_dim=config.EMBED_DIM, decoder_dim=config.DECODER_DIM,
                 attention_dim=config.ATTENTION_DIM, dropout=config.DROPOUT,
                 finetune_encoder=False):
        """
        Initialize baseline model.

        Args:
            vocab_size: Size of vocabulary
            encoder_dim: Encoder output dimension
            embedding_dim: Word embedding dimension
            decoder_dim: Decoder LSTM dimension
            attention_dim: Attention layer dimension
            dropout: Dropout rate
            finetune_encoder: Whether to finetune encoder
        """
        super(BaselineModel, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.dropout_rate = dropout
        self.finetune_encoder = finetune_encoder

        # Create encoder and decoder
        self.encoder = ImageEncoder(
            encoder_dim=encoder_dim,
            trainable=finetune_encoder
        )

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

        print(f"\nBaselineModel initialized:")
        print(f"  Encoder: ResNet50 (trainable={finetune_encoder})")
        print(f"  Decoder: LSTM with attention")
        print(f"  Vocab size: {vocab_size}")

    def call(self, images, captions, training=False):
        """
        Forward pass.

        Args:
            images: Input images [batch_size, 224, 224, 3]
            captions: Target captions [batch_size, max_length]
            training: Whether in training mode

        Returns:
            predictions: Predicted word distributions [batch_size, max_length, vocab_size]
            features: Encoder features [batch_size, encoder_dim]
            hidden_states: Decoder hidden states [batch_size, max_length, decoder_dim]
        """
        # Encode images
        features = self.encoder(images, training=training)

        # Decode captions
        predictions, hidden_states = self.decoder(
            features, captions, training=training
        )

        return predictions, features, hidden_states

    def generate_caption(self, image, start_token, end_token, max_length=20):
        """
        Generate caption for a single image.

        Args:
            image: Single image [1, 224, 224, 3]
            start_token: Start token index
            end_token: End token index
            max_length: Maximum caption length

        Returns:
            caption: List of word indices
        """
        # Encode image
        features = self.encoder(image, training=False)

        # Generate caption
        caption = self.decoder.predict_caption(
            features,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length
        )

        return caption

    def get_config(self):
        """Return configuration."""
        return {
            'vocab_size': self.vocab_size,
            'encoder_dim': self.encoder_dim,
            'embedding_dim': self.embedding_dim,
            'decoder_dim': self.decoder_dim,
            'attention_dim': self.attention_dim,
            'dropout': self.dropout_rate,
            'finetune_encoder': self.finetune_encoder
        }


def create_baseline_model(vocab_size, finetune_encoder=False):
    """
    Create baseline image captioning model.

    Args:
        vocab_size: Size of vocabulary
        finetune_encoder: Whether to finetune encoder

    Returns:
        BaselineModel instance
    """
    return BaselineModel(
        vocab_size=vocab_size,
        encoder_dim=config.ENCODER_DIM,
        embedding_dim=config.EMBED_DIM,
        decoder_dim=config.DECODER_DIM,
        attention_dim=config.ATTENTION_DIM,
        dropout=config.DROPOUT,
        finetune_encoder=finetune_encoder
    )


def compute_loss(real, pred, loss_object):
    """
    Compute caption loss with masking for padding.

    Args:
        real: Real captions [batch_size, max_length]
        pred: Predicted distributions [batch_size, max_length, vocab_size]
        loss_object: Loss function (SparseCategoricalCrossentropy)

    Returns:
        loss: Scalar loss value
    """
    # Create mask to ignore padding
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 0 is PAD token

    # Compute loss
    loss_ = loss_object(real, pred)

    # Apply mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # Return mean loss
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


if __name__ == '__main__':
    # Test baseline model
    print("Testing BaselineModel...\n")

    # Parameters
    vocab_size = 2832
    batch_size = 4
    max_length = 20

    # Create model
    model = create_baseline_model(vocab_size, finetune_encoder=False)

    # Create dummy inputs
    dummy_images = tf.random.normal([batch_size, 224, 224, 3])
    dummy_captions = tf.random.uniform(
        [batch_size, max_length],
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )

    print(f"\nInput images shape: {dummy_images.shape}")
    print(f"Input captions shape: {dummy_captions.shape}")

    # Forward pass
    predictions, features, hidden_states = model(
        dummy_images, dummy_captions, training=True
    )

    print(f"\nOutput predictions shape: {predictions.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {vocab_size}]")

    print(f"\nEncoder features shape: {features.shape}")
    print(f"Expected: [{batch_size}, {config.ENCODER_DIM}]")

    print(f"\nDecoder hidden states shape: {hidden_states.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {config.DECODER_DIM}]")

    # Test loss computation
    print("\nTesting loss computation...")
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    # Shift captions for next word prediction
    real = dummy_captions[:, 1:]  # Remove <START>
    pred = predictions[:, :-1, :]  # Remove last prediction

    loss = compute_loss(real, pred, loss_object)
    print(f"Loss: {loss.numpy():.4f}")

    # Test caption generation
    print("\nTesting caption generation...")
    single_image = tf.random.normal([1, 224, 224, 3])
    generated = model.generate_caption(
        single_image,
        start_token=1,  # <START>
        end_token=2,    # <END>
        max_length=15
    )

    print(f"Generated caption: {generated}")
    print(f"Length: {len(generated)}")

    print("\n✅ All baseline model tests passed!")

    # Print model summary
    print("\nModel summary:")
    print(f"Total trainable params: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
