"""
Adversary network for bias probing.
Used to measure how well race can be predicted from model features.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as config


class RaceAdversary(keras.Model):
    """
    Adversary network that tries to predict race from features.
    High accuracy = features contain racial information (biased).
    Low accuracy (~ 50%) = features are race-invariant (debiased).
    """

    def __init__(self, input_dim, hidden_dims=[512, 512, 512],
                 num_classes=config.NUM_RACE_CLASSES, dropout=0.3):
        """
        Initialize adversary.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of race classes (2: Light/Dark)
            dropout: Dropout rate
        """
        super(RaceAdversary, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Build MLP
        self.layers_list = []

        # Input layer
        self.layers_list.append(layers.BatchNormalization())
        self.layers_list.append(layers.Dense(hidden_dims[0]))
        self.layers_list.append(layers.LeakyReLU(alpha=0.2))
        self.layers_list.append(layers.Dropout(dropout))

        # Hidden layers
        for dim in hidden_dims[1:]:
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(layers.Dense(dim))
            self.layers_list.append(layers.LeakyReLU(alpha=0.2))
            self.layers_list.append(layers.Dropout(dropout))

        # Output layer
        self.layers_list.append(layers.Dense(num_classes))

        print(f"RaceAdversary initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Output classes: {num_classes}")
        print(f"  Dropout: {dropout}")

    def call(self, features, training=False):
        """
        Forward pass.

        Args:
            features: Input features [batch_size, input_dim]
            training: Whether in training mode

        Returns:
            race_logits: Race predictions [batch_size, num_classes]
        """
        x = features

        for layer in self.layers_list:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return x

    def get_config(self):
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate
        }


def create_encoder_adversary(input_dim=config.ENCODER_DIM):
    """
    Create adversary for encoder features.

    Args:
        input_dim: Encoder feature dimension

    Returns:
        RaceAdversary instance
    """
    return RaceAdversary(
        input_dim=input_dim,
        hidden_dims=[512, 512, 512],
        num_classes=config.NUM_RACE_CLASSES,
        dropout=0.3
    )


def create_decoder_adversary(input_dim=config.DECODER_DIM):
    """
    Create adversary for decoder hidden states.

    Args:
        input_dim: Decoder hidden state dimension

    Returns:
        RaceAdversary instance
    """
    return RaceAdversary(
        input_dim=input_dim,
        hidden_dims=[256, 256],
        num_classes=config.NUM_RACE_CLASSES,
        dropout=0.3
    )


if __name__ == '__main__':
    # Test adversary
    print("Testing RaceAdversary...\n")

    # Test encoder adversary
    print("Creating encoder adversary...")
    encoder_adv = create_encoder_adversary(input_dim=2048)

    batch_size = 4
    encoder_features = tf.random.normal([batch_size, 2048])

    print(f"\nInput shape: {encoder_features.shape}")

    encoder_preds = encoder_adv(encoder_features, training=True)
    print(f"Output shape: {encoder_preds.shape}")
    print(f"Expected: [{batch_size}, {config.NUM_RACE_CLASSES}]")

    assert encoder_preds.shape == (batch_size, config.NUM_RACE_CLASSES)

    # Test decoder adversary
    print("\n\nCreating decoder adversary...")
    decoder_adv = create_decoder_adversary(input_dim=512)

    decoder_hidden = tf.random.normal([batch_size, 512])
    decoder_preds = decoder_adv(decoder_hidden, training=True)

    print(f"\nInput shape: {decoder_hidden.shape}")
    print(f"Output shape: {decoder_preds.shape}")
    print(f"Expected: [{batch_size}, {config.NUM_RACE_CLASSES}]")

    assert decoder_preds.shape == (batch_size, config.NUM_RACE_CLASSES)

    print("\n✅ All adversary tests passed!")

    # Test with softmax
    print("\nTesting with softmax...")
    probs = tf.nn.softmax(encoder_preds, axis=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {tf.reduce_sum(probs, axis=-1).numpy()}")

    print("\n✅ RaceAdversary implementation complete!")
