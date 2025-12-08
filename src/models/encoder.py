"""
Vision encoder using ResNet50 for image feature extraction.
"""

import tensorflow as tf
from tensorflow import keras
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as config


class ImageEncoder(keras.Model):
    """
    Image encoder using pretrained ResNet50.
    Extracts visual features from images for caption generation.
    """

    def __init__(self, encoder_dim=config.ENCODER_DIM, trainable=False):
        """
        Initialize encoder.

        Args:
            encoder_dim: Output dimension of encoder features
            trainable: Whether to finetune encoder weights
        """
        super(ImageEncoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.trainable = trainable

        # Load pretrained ResNet50 (without top classification layer)
        self.resnet = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            pooling='avg'  # Global average pooling
        )

        # Set trainability
        self.resnet.trainable = trainable

        # Output should be [batch_size, encoder_dim]
        # ResNet50 with avg pooling outputs 2048-dim features
        assert self.resnet.output_shape[-1] == encoder_dim, \
            f"ResNet50 output dim {self.resnet.output_shape[-1]} != encoder_dim {encoder_dim}"

        print(f"ImageEncoder initialized:")
        print(f"  Base model: ResNet50")
        print(f"  Output dim: {encoder_dim}")
        print(f"  Trainable: {trainable}")
        print(f"  Total params: {self.resnet.count_params():,}")

    def call(self, images, training=False):
        """
        Forward pass.

        Args:
            images: Input images [batch_size, 224, 224, 3]
            training: Whether in training mode

        Returns:
            Image features [batch_size, encoder_dim]
        """
        # Extract features using ResNet50
        # Input: [B, 224, 224, 3]
        # Output: [B, 2048] (global average pooling)
        features = self.resnet(images, training=training and self.trainable)

        return features

    def get_config(self):
        """Return configuration for serialization."""
        return {
            'encoder_dim': self.encoder_dim,
            'trainable': self.trainable
        }


def create_encoder(encoder_dim=config.ENCODER_DIM, trainable=False):
    """
    Create image encoder.

    Args:
        encoder_dim: Output dimension
        trainable: Whether to finetune encoder

    Returns:
        ImageEncoder instance
    """
    return ImageEncoder(encoder_dim=encoder_dim, trainable=trainable)


if __name__ == '__main__':
    # Test encoder
    print("Testing ImageEncoder...\n")

    # Create encoder
    encoder = create_encoder(trainable=False)

    # Create dummy input
    batch_size = 4
    dummy_images = tf.random.normal([batch_size, 224, 224, 3])

    print(f"\nInput shape: {dummy_images.shape}")

    # Forward pass
    features = encoder(dummy_images, training=False)

    print(f"Output shape: {features.shape}")
    print(f"Expected shape: [{batch_size}, {config.ENCODER_DIM}]")

    assert features.shape == (batch_size, config.ENCODER_DIM), \
        f"Output shape mismatch: {features.shape}"

    print("\n✅ ImageEncoder test passed!")

    # Test with training mode
    print("\nTesting training mode...")
    features_train = encoder(dummy_images, training=True)
    print(f"Training mode output shape: {features_train.shape}")

    # Check trainable variables
    print(f"\nTrainable variables: {len(encoder.trainable_variables)}")
    print(f"Non-trainable variables: {len(encoder.non_trainable_variables)}")

    print("\n✅ All encoder tests passed!")
