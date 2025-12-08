"""
Dual-location adversarial debiasing model.
Applies gradient reversal at both encoder features and decoder hidden states.
"""

import tensorflow as tf
from tensorflow import keras
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoder import ImageEncoder
from models.decoder import CaptionDecoder
from models.adversary import create_encoder_adversary, create_decoder_adversary
from models.gradient_reversal import GradientReversalLayer
import utils.config as config


class DualAdversarialModel(keras.Model):
    """
    Image captioning model with dual-location adversarial debiasing.
    """
    
    def __init__(self, vocab_size, 
                 encoder_dim=config.ENCODER_DIM,
                 embedding_dim=config.EMBED_DIM, 
                 decoder_dim=config.DECODER_DIM,
                 attention_dim=config.ATTENTION_DIM, 
                 dropout=config.DROPOUT,
                 lambda_v=0.5, 
                 lambda_d=0.5,
                 finetune_encoder=False):
        """
        Initialize dual adversarial model.
        
        Args:
            vocab_size: Size of vocabulary
            encoder_dim: Encoder output dimension
            embedding_dim: Word embedding dimension
            decoder_dim: Decoder LSTM dimension
            attention_dim: Attention layer dimension
            dropout: Dropout rate
            lambda_v: Gradient reversal weight for encoder adversary
            lambda_d: Gradient reversal weight for decoder adversary
            finetune_encoder: Whether to finetune encoder
        """
        super(DualAdversarialModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.dropout_rate = dropout
        self.lambda_v = lambda_v
        self.lambda_d = lambda_d
        self.finetune_encoder = finetune_encoder
        
        # Main model components
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
        
        # Gradient reversal layers
        self.grl_encoder = GradientReversalLayer(lambda_val=lambda_v)
        self.grl_decoder = GradientReversalLayer(lambda_val=lambda_d)
        
        # Adversary networks
        self.adversary_encoder = create_encoder_adversary(input_dim=encoder_dim)
        self.adversary_decoder = create_decoder_adversary(input_dim=decoder_dim)
        
        print(f"\nDualAdversarialModel initialized:")
        print(f"  Encoder: ResNet50 (trainable={finetune_encoder})")
        print(f"  Decoder: LSTM with attention")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Lambda encoder (λ_v): {lambda_v}")
        print(f"  Lambda decoder (λ_d): {lambda_d}")
        print(f"  Adversary encoder: {encoder_dim} → 512 → 512 → 512 → 2")
        print(f"  Adversary decoder: {decoder_dim} → 256 → 256 → 2")
    
    def call(self, images, captions, training=False):
        """
        Forward pass with dual adversarial debiasing.
        
        Args:
            images: Input images [batch_size, 224, 224, 3]
            captions: Target captions [batch_size, max_length]
            training: Whether in training mode
        
        Returns:
            predictions: Predicted word distributions [batch_size, max_length, vocab_size]
            encoder_features: Encoder features [batch_size, encoder_dim]
            decoder_hiddens: Decoder hidden states [batch_size, max_length, decoder_dim]
            race_pred_encoder: Race predictions from encoder features [batch_size, 2]
            race_pred_decoder: Race predictions from decoder hiddens [batch_size, 2]
        """
        # Encode images
        encoder_features = self.encoder(images, training=training)
        
        # Decode captions
        predictions, decoder_hiddens = self.decoder(
            encoder_features, captions, training=training
        )
        
        # Apply gradient reversal and adversarial classification
        # Encoder adversary
        reversed_encoder = self.grl_encoder(encoder_features)
        race_pred_encoder = self.adversary_encoder(reversed_encoder, training=training)
        
        # Decoder adversary
        final_decoder_hidden = decoder_hiddens[:, -1, :]
        
        reversed_decoder = self.grl_decoder(final_decoder_hidden)
        race_pred_decoder = self.adversary_decoder(reversed_decoder, training=training)
        
        return (predictions, encoder_features, decoder_hiddens, race_pred_encoder, race_pred_decoder)
    
    def call_with_temporal_adversary(self, images, captions, training=False):
        """
        Alternative forward pass with temporal averaging for decoder adversary.
        
        Instead of using only the final hidden state, this averages predictions
        across all timesteps for more robust debiasing.
        
        Args:
            images: Input images [batch_size, 224, 224, 3]
            captions: Target captions [batch_size, max_length]
            training: Whether in training mode
        
        Returns:
            Same as call(), but race_pred_decoder is averaged over timesteps
        """
        # Encode images
        encoder_features = self.encoder(images, training=training)
        
        # Decode captions
        predictions, decoder_hiddens = self.decoder(
            encoder_features, captions, training=training
        )
        
        # Encoder adversary
        reversed_encoder = self.grl_encoder(encoder_features)
        race_pred_encoder = self.adversary_encoder(reversed_encoder, training=training)
        
        # Decoder adversary with temporal averaging
        # Apply adversary to each timestep and average
        max_length = tf.shape(decoder_hiddens)[1]
        
        race_preds_list = []
        for t in range(config.MAX_CAPTION_LENGTH):
            hidden_t = decoder_hiddens[:, t, :]
            reversed_t = self.grl_decoder(hidden_t)
            pred_t = self.adversary_decoder(reversed_t, training=training)
            race_preds_list.append(pred_t)
        
        # Average predictions across timesteps
        race_pred_decoder = tf.reduce_mean(
            tf.stack(race_preds_list, axis=1), axis=1
        )
        
        return (predictions, encoder_features, decoder_hiddens,
                race_pred_encoder, race_pred_decoder)
    
    def generate_caption(self, image, start_token, end_token, max_length=20):
        """
        Generate caption for a single image (inference mode).
        
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
    
    def update_lambda_values(self, lambda_v=None, lambda_d=None):
        """
        Update gradient reversal lambda values during training.
        
        Args:
            lambda_v: New value for encoder adversary (None = keep current)
            lambda_d: New value for decoder adversary (None = keep current)
        """
        if lambda_v is not None:
            self.lambda_v = lambda_v
            self.grl_encoder.lambda_val = lambda_v
            print(f"Updated λ_v to {lambda_v}")
        
        if lambda_d is not None:
            self.lambda_d = lambda_d
            self.grl_decoder.lambda_val = lambda_d
            print(f"Updated λ_d to {lambda_d}")
    
    def get_config(self):
        """Return configuration."""
        return {
            'vocab_size': self.vocab_size,
            'encoder_dim': self.encoder_dim,
            'embedding_dim': self.embedding_dim,
            'decoder_dim': self.decoder_dim,
            'attention_dim': self.attention_dim,
            'dropout': self.dropout_rate,
            'lambda_v': self.lambda_v,
            'lambda_d': self.lambda_d,
            'finetune_encoder': self.finetune_encoder
        }


def create_dual_adversarial_model(vocab_size, lambda_v=0.5, lambda_d=0.5, 
                                  finetune_encoder=False, use_temporal=False):
    """
    Create dual-location adversarial debiasing model.
    
    Args:
        vocab_size: Size of vocabulary
        lambda_v: Gradient reversal weight for encoder
        lambda_d: Gradient reversal weight for decoder
        finetune_encoder: Whether to finetune encoder
        use_temporal: Whether to use temporal averaging (not used in construction)
    
    Returns:
        DualAdversarialModel instance
    """
    return DualAdversarialModel(
        vocab_size=vocab_size,
        encoder_dim=config.ENCODER_DIM,
        embedding_dim=config.EMBED_DIM,
        decoder_dim=config.DECODER_DIM,
        attention_dim=config.ATTENTION_DIM,
        dropout=config.DROPOUT,
        lambda_v=lambda_v,
        lambda_d=lambda_d,
        finetune_encoder=finetune_encoder
    )


def compute_dual_loss(real_captions, pred_captions, race_labels,
                     race_pred_encoder, race_pred_decoder,
                     caption_loss_fn, lambda_v, lambda_d):
    """
    Compute combined loss for dual adversarial training.
    
    Loss = L_caption - λ_v * L_adv_encoder - λ_d * L_adv_decoder
    
    Args:
        real_captions: Ground truth captions [batch_size, max_length]
        pred_captions: Predicted captions [batch_size, max_length, vocab_size]
        race_labels: Race labels [batch_size]
        race_pred_encoder: Encoder adversary predictions [batch_size, 2]
        race_pred_decoder: Decoder adversary predictions [batch_size, 2]
        caption_loss_fn: Caption loss function
        lambda_v: Weight for encoder adversary loss
        lambda_d: Weight for decoder adversary loss
    
    Returns:
        total_loss: Combined loss for main model
        caption_loss: Caption generation loss
        adv_loss_encoder: Encoder adversary loss
        adv_loss_decoder: Decoder adversary loss
    """
    # Create mask for padding
    mask = tf.math.logical_not(tf.math.equal(real_captions, 0))
    
    # Caption loss
    loss_ = caption_loss_fn(real_captions, pred_captions)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    caption_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    
    # Adversary losses
    adv_loss_encoder = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            race_labels, race_pred_encoder, from_logits=True
        )
    )
    
    adv_loss_decoder = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            race_labels, race_pred_decoder, from_logits=True
        )
    )
    
    # Combined loss
    total_loss = caption_loss - lambda_v * adv_loss_encoder - lambda_d * adv_loss_decoder
    
    return total_loss, caption_loss, adv_loss_encoder, adv_loss_decoder


if __name__ == '__main__':
    # Test dual adversarial model
    print("Testing DualAdversarialModel...\n")
    
    # Parameters
    vocab_size = 2832
    batch_size = 4
    max_length = 20
    
    # Create model
    model = create_dual_adversarial_model(
        vocab_size=vocab_size,
        lambda_v=0.5,
        lambda_d=0.5,
        finetune_encoder=False
    )
    
    # Create dummy inputs
    dummy_images = tf.random.normal([batch_size, 224, 224, 3])
    dummy_captions = tf.random.uniform(
        [batch_size, max_length],
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    dummy_race_labels = tf.random.uniform(
        [batch_size],
        minval=0,
        maxval=2,
        dtype=tf.int32
    )
    
    print(f"Input images shape: {dummy_images.shape}")
    print(f"Input captions shape: {dummy_captions.shape}")
    print(f"Race labels shape: {dummy_race_labels.shape}")
    
    # Test forward pass
    print("\n\nTesting standard forward pass...")
    (predictions, encoder_features, decoder_hiddens,
     race_pred_encoder, race_pred_decoder) = model(
        dummy_images, dummy_captions, training=True
    )
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {vocab_size}]")
    
    print(f"\nEncoder features shape: {encoder_features.shape}")
    print(f"Expected: [{batch_size}, {config.ENCODER_DIM}]")
    
    print(f"\nDecoder hiddens shape: {decoder_hiddens.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {config.DECODER_DIM}]")
    
    print(f"\nRace pred encoder shape: {race_pred_encoder.shape}")
    print(f"Expected: [{batch_size}, 2]")
    
    print(f"\nRace pred decoder shape: {race_pred_decoder.shape}")
    print(f"Expected: [{batch_size}, 2]")
    
    # Test loss computation
    print("\n\nTesting loss computation...")
    caption_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    
    # Shift for next word prediction
    real = dummy_captions[:, 1:]
    pred = predictions[:, :-1, :]
    
    total_loss, cap_loss, adv_loss_v, adv_loss_d = compute_dual_loss(
        real, pred, dummy_race_labels,
        race_pred_encoder, race_pred_decoder,
        caption_loss_fn, lambda_v=0.5, lambda_d=0.5
    )
    
    print(f"Total loss: {total_loss.numpy():.4f}")
    print(f"Caption loss: {cap_loss.numpy():.4f}")
    print(f"Encoder adversary loss: {adv_loss_v.numpy():.4f}")
    print(f"Decoder adversary loss: {adv_loss_d.numpy():.4f}")
    
    # Test temporal adversary
    print("\n\nTesting temporal adversary forward pass...")
    (predictions2, encoder_features2, decoder_hiddens2,
     race_pred_encoder2, race_pred_decoder2) = model.call_with_temporal_adversary(
        dummy_images, dummy_captions, training=True
    )
    
    print(f"Race pred decoder (temporal) shape: {race_pred_decoder2.shape}")
    print(f"Expected: [{batch_size}, 2]")
    
    # Test lambda updates
    print("\n\nTesting lambda value updates...")
    model.update_lambda_values(lambda_v=0.7, lambda_d=0.3)
    
    # Test caption generation
    print("\n\nTesting caption generation...")
    single_image = tf.random.normal([1, 224, 224, 3])
    generated = model.generate_caption(
        single_image,
        start_token=1,
        end_token=2,
        max_length=15
    )
    
    print(f"Generated caption: {generated}")
    print(f"Length: {len(generated)}")
    
    print("\nAll dual adversarial model tests passed!")
    
    # Print model summary
    print("\nModel components:")
    print(f"Main model trainable params: {sum([tf.size(v).numpy() for v in model.encoder.trainable_variables + model.decoder.trainable_variables]):,}")
    print(f"Adversary encoder params: {sum([tf.size(v).numpy() for v in model.adversary_encoder.trainable_variables]):,}")
    print(f"Adversary decoder params: {sum([tf.size(v).numpy() for v in model.adversary_decoder.trainable_variables]):,}")