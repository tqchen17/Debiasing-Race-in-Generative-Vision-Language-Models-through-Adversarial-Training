"""
Caption decoder with LSTM and Bahdanau attention mechanism.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as config


class BahdanauAttention(keras.layers.Layer):
    """
    Bahdanau attention mechanism.
    Computes attention weights over encoder features based on decoder state.
    """

    def __init__(self, units):
        """
        Initialize attention layer.

        Args:
            units: Dimension of attention layer
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)  # For encoder features
        self.W2 = layers.Dense(units)  # For decoder hidden state
        self.V = layers.Dense(1)       # For attention scores

    def call(self, features, hidden):
        """
        Compute attention.

        Args:
            features: Encoder features [batch_size, encoder_dim]
            hidden: Decoder hidden state [batch_size, decoder_dim]

        Returns:
            context_vector: Weighted sum of features [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, 1]
        """
        # Expand features for broadcasting
        # features shape: [batch_size, encoder_dim]
        # We need to expand to [batch_size, 1, encoder_dim] for attention
        features_expanded = tf.expand_dims(features, 1)

        # hidden shape: [batch_size, decoder_dim]
        # Expand to [batch_size, 1, decoder_dim]
        hidden_with_time = tf.expand_dims(hidden, 1)

        # Calculate attention scores
        # score = V * tanh(W1(features) + W2(hidden))
        score = self.V(tf.nn.tanh(
            self.W1(features_expanded) + self.W2(hidden_with_time)
        ))  # [batch_size, 1, 1]

        # Attention weights (no need for softmax with single feature vector)
        # For image captioning with global pooled features, we have single feature vector
        # So attention_weights is just sigmoid of score
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch_size, 1, 1]

        # Context vector: weighted sum of features
        context_vector = attention_weights * features_expanded  # [batch_size, 1, encoder_dim]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_size, encoder_dim]

        return context_vector, attention_weights


class CaptionDecoder(keras.Model):
    """
    LSTM-based caption decoder with attention.
    """

    def __init__(self, vocab_size, embedding_dim=config.EMBED_DIM,
                 decoder_dim=config.DECODER_DIM, attention_dim=config.ATTENTION_DIM,
                 dropout=config.DROPOUT):
        """
        Initialize decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            decoder_dim: Dimension of LSTM hidden state
            attention_dim: Dimension of attention layer
            dropout: Dropout rate
        """
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.dropout_rate = dropout

        # Word embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )

        # Attention mechanism
        self.attention = BahdanauAttention(attention_dim)

        # LSTM cell
        self.lstm_cell = layers.LSTMCell(decoder_dim)

        # Dropout
        self.dropout = layers.Dropout(dropout)

        # Output layers
        self.fc1 = layers.Dense(decoder_dim, activation='relu')
        self.fc_out = layers.Dense(vocab_size)

        print(f"CaptionDecoder initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Decoder dim: {decoder_dim}")
        print(f"  Attention dim: {attention_dim}")
        print(f"  Dropout: {dropout}")

    def call(self, features, captions, training=False):
        """
        Forward pass for training.

        Args:
            features: Encoder features [batch_size, encoder_dim]
            captions: Target captions [batch_size, max_length]
            training: Whether in training mode

        Returns:
            predictions: Predicted word distributions [batch_size, max_length, vocab_size]
            hidden_states: LSTM hidden states for each timestep [batch_size, max_length, decoder_dim]
        """
        batch_size = tf.shape(features)[0]

        # Initialize LSTM state with zeros
        state = [tf.zeros([batch_size, self.decoder_dim]),
                 tf.zeros([batch_size, self.decoder_dim])]

        # Embed captions
        # captions shape: [batch_size, max_length]
        embeddings = self.embedding(captions)  # [batch_size, max_length, embedding_dim]
        embeddings = self.dropout(embeddings, training=training)

        # Lists to store outputs
        predictions_list = []
        hidden_states_list = []

        # Loop through each timestep
        # Use constant max_length instead of tf.shape for @tf.function compatibility
        max_length = config.MAX_CAPTION_LENGTH

        for t in range(max_length):
            # Get embedding for current timestep
            x = embeddings[:, t, :]  # [batch_size, embedding_dim]

            # Attention
            context, attention_weights = self.attention(features, state[0])

            # Concatenate context and embedding
            lstm_input = tf.concat([context, x], axis=-1)  # [batch_size, encoder_dim + embedding_dim]

            # LSTM step
            output, state = self.lstm_cell(lstm_input, state, training=training)
            # output: [batch_size, decoder_dim]

            hidden_states_list.append(state[0])  # Store hidden state

            # Apply dropout
            output = self.dropout(output, training=training)

            # Predict word distribution
            output = self.fc1(output)
            output = self.dropout(output, training=training)
            preds = self.fc_out(output)  # [batch_size, vocab_size]

            predictions_list.append(preds)

        # Stack predictions
        predictions = tf.stack(predictions_list, axis=1)  # [batch_size, max_length, vocab_size]
        hidden_states = tf.stack(hidden_states_list, axis=1)  # [batch_size, max_length, decoder_dim]

        return predictions, hidden_states

    def predict_caption(self, features, start_token, end_token, max_length=20):
        """
        Generate caption using greedy decoding.

        Args:
            features: Encoder features [1, encoder_dim] (single image)
            start_token: Start token index
            end_token: End token index
            max_length: Maximum caption length

        Returns:
            generated_caption: List of word indices
        """
        # Initialize state
        state = [tf.zeros([1, self.decoder_dim]),
                 tf.zeros([1, self.decoder_dim])]

        # Start with start token
        current_token = tf.expand_dims([start_token], 0)  # [1, 1]

        generated_caption = []

        for t in range(max_length):
            # Embed current token
            x = self.embedding(current_token)  # [1, 1, embedding_dim]
            x = tf.squeeze(x, axis=1)  # [1, embedding_dim]

            # Attention
            context, _ = self.attention(features, state[0])

            # Concatenate
            lstm_input = tf.concat([context, x], axis=-1)

            # LSTM step
            output, state = self.lstm_cell(lstm_input, state, training=False)

            # Predict next word
            output = self.fc1(output)
            preds = self.fc_out(output)  # [1, vocab_size]

            # Greedy selection
            predicted_id = tf.argmax(preds, axis=-1)[0].numpy()

            # Add to caption
            generated_caption.append(predicted_id)

            # Stop if end token
            if predicted_id == end_token:
                break

            # Update current token
            current_token = tf.expand_dims([predicted_id], 0)

        return generated_caption

    def get_config(self):
        """Return configuration."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'decoder_dim': self.decoder_dim,
            'attention_dim': self.attention_dim,
            'dropout': self.dropout_rate
        }


def create_decoder(vocab_size, embedding_dim=config.EMBED_DIM,
                  decoder_dim=config.DECODER_DIM, attention_dim=config.ATTENTION_DIM,
                  dropout=config.DROPOUT):
    """
    Create caption decoder.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        decoder_dim: LSTM dimension
        attention_dim: Attention dimension
        dropout: Dropout rate

    Returns:
        CaptionDecoder instance
    """
    return CaptionDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test decoder
    print("Testing CaptionDecoder...\n")

    # Parameters
    vocab_size = 2832  # From our vocabulary
    batch_size = 4
    max_length = 20
    encoder_dim = 2048

    # Create decoder
    decoder = create_decoder(vocab_size)

    # Create dummy inputs
    dummy_features = tf.random.normal([batch_size, encoder_dim])
    dummy_captions = tf.random.uniform(
        [batch_size, max_length],
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )

    print(f"Features shape: {dummy_features.shape}")
    print(f"Captions shape: {dummy_captions.shape}")

    # Forward pass
    predictions, hidden_states = decoder(dummy_features, dummy_captions, training=True)

    print(f"\nOutput predictions shape: {predictions.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {vocab_size}]")

    print(f"\nHidden states shape: {hidden_states.shape}")
    print(f"Expected: [{batch_size}, {max_length}, {config.DECODER_DIM}]")

    assert predictions.shape == (batch_size, max_length, vocab_size)
    assert hidden_states.shape == (batch_size, max_length, config.DECODER_DIM)

    print("\n✅ CaptionDecoder forward pass test passed!")

    # Test caption generation
    print("\nTesting caption generation...")
    single_feature = tf.random.normal([1, encoder_dim])
    generated = decoder.predict_caption(
        single_feature,
        start_token=1,  # <START>
        end_token=2,    # <END>
        max_length=15
    )

    print(f"Generated caption indices: {generated}")
    print(f"Length: {len(generated)}")

    print("\n✅ All decoder tests passed!")
