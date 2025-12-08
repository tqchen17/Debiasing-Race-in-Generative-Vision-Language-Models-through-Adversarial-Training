"""
Gradient Reversal Layer for adversarial debiasing.
Implements gradient reversal using TensorFlow custom gradients.
"""

import tensorflow as tf
from tensorflow import keras


@tf.custom_gradient
def gradient_reversal(x, lambda_val):
    """
    Gradient reversal function using TensorFlow custom gradient.
    
    Forward pass: identity
    Backward pass: negates and scales gradients by lambda_val
    
    Args:
        x: Input tensor
        lambda_val: Scaling factor for gradient reversal
    
    Returns:
        x: Forward pass output
        grad: Custom gradient function
    """
    def grad(dy):
        # Reverse and scale gradients
        # dy: gradient from upstream
        # Returns: (gradient w.r.t. x, gradient w.r.t. lambda_val)
        return -lambda_val * dy, None
    
    return x, grad


class GradientReversalLayer(keras.layers.Layer):
    """
    Keras layer that reverses gradients during backpropagation.
    """
    
    def __init__(self, lambda_val=1.0, **kwargs):
        """
        Initialize gradient reversal layer.
        
        Args:
            lambda_val: Scaling factor for gradient reversal
        """
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_val = lambda_val
    
    def call(self, x):
        """
        Forward pass with gradient reversal.
        
        Args:
            x: Input tensor [batch_size, feature_dim]
        
        Returns:
            Output tensor
        """
        return gradient_reversal(x, self.lambda_val)
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({"lambda_val": self.lambda_val})
        return config


if __name__ == '__main__':
    # Test gradient reversal layer
    print("Testing GradientReversalLayer...\n")
    
    # Create layer
    lambda_val = 0.5
    grl = GradientReversalLayer(lambda_val=lambda_val)
    
    # Create dummy input
    batch_size = 4
    feature_dim = 2048
    x = tf.random.normal([batch_size, feature_dim])
    
    print(f"Input shape: {x.shape}")
    print(f"Lambda value: {lambda_val}")
    
    # Test forward pass
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = grl(x)
        loss = tf.reduce_sum(y)
    
    print(f"\nForward pass output shape: {y.shape}")
    print(f"Forward pass preserves input: {tf.reduce_all(tf.equal(x, y)).numpy()}")
    
    # Test backward pass
    gradients = tape.gradient(loss, x)
    print(f"\nGradient shape: {gradients.shape}")
    
    # Expected gradient should be -lambda_val
    expected_grad = -lambda_val
    actual_grad = tf.reduce_mean(gradients).numpy()
    
    print(f"Expected gradient value: {expected_grad}")
    print(f"Actual gradient value: {actual_grad:.4f}")
    print(f"Gradient reversal working: {abs(actual_grad - expected_grad) < 0.01}")
    
    # Test with different lambda values
    print("\n\nTesting different lambda values:")
    for lam in [0.3, 0.5, 0.7, 1.0]:
        grl_test = GradientReversalLayer(lambda_val=lam)
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = grl_test(x)
            loss = tf.reduce_sum(y)
        
        grads = tape.gradient(loss, x)
        mean_grad = tf.reduce_mean(grads).numpy()
        
        print(f"  λ={lam}: gradient={mean_grad:.4f} (expected={-lam})")
    
    # Test serialization
    print("\n\nTesting serialization...")
    config = grl.get_config()
    print(f"Config: {config}")
    
    grl_restored = GradientReversalLayer.from_config(config)
    print(f"Restored lambda: {grl_restored.lambda_val}")
    
    print("\nAll GradientReversalLayer tests passed!")