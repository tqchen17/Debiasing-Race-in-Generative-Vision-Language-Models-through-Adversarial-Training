# Implementation Plan: Debiasing Race in Generative Vision-Language Models

## Project Overview

Extending adversarial debiasing from classification tasks to **image captioning for racial bias mitigation**. The key innovation is applying gradient reversal at two locations: encoder features and decoder hidden states.

### Team Members
- Vignesh Peddi (vpeddi1)
- Thomas Chen (tqchen)
- Joao Gabriel Pepato de Oliveira (jpdeoliv)
- Kanpat Vesessook (kvesesso)

## Architecture

```
Image → Vision Encoder (CLIP/ViT/ResNet) → Visual Features [h_v]
                                              ↓              ↓
                                        Adversary_v    Decoder (Tokens)
                                     (Gradient Reversal)      ↓
                                                         Hidden [h_d]
                                                       ↓          ↓
                                                  Adversary_d  Output
                                               (Gradient Reversal)
```

### Key Differences from Original Paper
- **Original:** Classification task (gender prediction from images)
- **Our Work:** Generative task (image captioning with race debiasing)
- **Innovation:** Dual-location gradient reversal (encoder + decoder)

## Current Status

### Completed
- ✅ Data preprocessing script (`preprocess.py`)
- ✅ Princeton racial bias dataset (28,315 images with skin color annotations)
- ✅ Balanced Datasets repository code (gradient reversal reference)
- ✅ Dataset: COCO images + Princeton annotations merged

### In Progress
- ⏳ Running preprocessing to generate MASTER_TRAIN.csv and MASTER_VAL.csv

## Implementation Phases

### Phase 1: Foundation (Base Goal)

#### 1.1 Data Infrastructure
**Files:** `src/data/dataset.py`, `src/data/data_generator.py`

**Tasks:**
- [ ] Verify MASTER_TRAIN.csv and MASTER_VAL.csv structure
  - Columns: `image_path`, `caption_text`, `race_label`, `bb_skin`, `id`
- [ ] Implement TensorFlow data pipeline (tf.data.Dataset) for image-caption-race triplets
- [ ] Create balanced batch sampling (equal race distribution)
- [ ] Data augmentation pipeline using tf.image

**Expected Output:**
```python
# Example batch (TensorFlow tensors)
{
    'image': tf.Tensor([B, 224, 224, 3], dtype=tf.float32),
    'caption': tf.Tensor([B, max_len], dtype=tf.int32),  # tokenized
    'race_label': tf.Tensor([B], dtype=tf.int32),  # 0=Light, 1=Dark
    'caption_lengths': tf.Tensor([B], dtype=tf.int32)
}
```

#### 1.2 Baseline Model (No Debiasing)
**Files:** `src/models/encoder.py`, `src/models/decoder.py`, `src/models/baseline_model.py`

**Tasks:**
- [ ] Vision encoder: tf.keras.applications.ResNet50 (pretrained on ImageNet, frozen initially)
- [ ] Caption decoder: LSTM (512-dim hidden) with Bahdanau attention
- [ ] Standard sparse categorical cross-entropy loss for caption generation
- [ ] Training script: `src/training/baseline_train.py`
- [ ] Evaluation: BLEU-4, ROUGE-L on validation set

**Baseline Hyperparameters:**
```python
encoder = 'resnet50'  # tf.keras.applications.ResNet50
encoder_dim = 2048  # ResNet50 output dimension
decoder_dim = 512
embed_dim = 300
attention_dim = 512
dropout = 0.5
lr_encoder = 1e-4  # if finetuning
lr_decoder = 1e-3
batch_size = 32
num_epochs = 20
```

**Success Criteria:**
- BLEU-4 ≥ 0.25 (reasonable caption quality)
- Model converges within 20 epochs
- Establishes baseline for comparison

#### 1.3 Single-Location Adversarial Debiasing (Encoder)
**Files:** `src/models/gradient_reversal.py`, `src/models/adversary.py`, `src/models/adv_model_encoder.py`

**Tasks:**
- [ ] Implement `GradientReversalLayer` as a custom Keras layer
  - Use `@tf.custom_gradient` decorator for gradient reversal
  - Reference concept from: `Balanced-Datasets-Are-Not-Enough-master/object_multilabel/adv/model.py:190-200`
- [ ] Race classifier (Adversary_v) on encoder features
  - Architecture: 3-layer MLP with BatchNormalization + LeakyReLU
  - Input: encoder features (2048-dim for ResNet50)
  - Output: 2-class (Light/Dark)
- [ ] Combined training loss:
  ```python
  L_total = L_caption - λ_v * L_adversary_v
  ```
- [ ] Training script: `src/training/adv_train_encoder.py`

**Adversary Architecture:**
```python
# Keras Sequential model
AdversaryNetwork:
  Dense(2048 → 512) + BatchNormalization + LeakyReLU(alpha=0.2)
  Dense(512 → 512) + BatchNormalization + LeakyReLU(alpha=0.2)
  Dense(512 → 512) + BatchNormalization + LeakyReLU(alpha=0.2)
  Dense(512 → 2, activation='softmax')  # race prediction
```

**Hyperparameters:**
```python
lambda_v = 0.5  # start here, tune on validation
adversary_lr = 1e-3
hid_size = 512
```

**Success Criteria:**
- Adversary_v accuracy drops toward 50% (random guessing)
- Caption BLEU ≥ 95% of baseline
- Visual encoder learns race-invariant features

#### 1.4 Initial Evaluation
**Files:** `src/evaluation/caption_metrics.py`, `src/evaluation/bias_metrics.py`

**Tasks:**
- [ ] Implement BLEU-1,2,3,4 (using `nltk.translate.bleu_score`)
- [ ] Implement ROUGE-L (using `rouge-score` library)
- [ ] Adversary probing accuracy on encoder features
- [ ] Compare: Baseline vs Single-Location Debiasing
  - Caption quality (BLEU/ROUGE)
  - Adversary accuracy
  - Per-race caption quality (fairness metric)

**Evaluation Metrics:**
```python
Metrics to Report:
1. Caption Quality:
   - BLEU-4
   - ROUGE-L
   - CIDEr (optional)

2. Bias Metrics:
   - Adversary accuracy (goal: ~50%)
   - Per-race BLEU (Light vs Dark)
   - Race word frequency in captions
```

---

### Phase 2: Core Innovation (Target Goal)

#### 2.1 Dual-Location Gradient Reversal
**Files:** `src/models/adv_model_dual.py`, `src/training/adv_train_dual.py`

**Tasks:**
- [ ] Add second adversary (Adversary_d) on decoder hidden states
  - Option A: Apply on final decoder hidden state
  - Option B: Apply at each timestep, average adversarial loss
- [ ] Implement dual gradient reversal:
  ```python
  # Encoder features
  grl_v = GradientReversalLayer(lambda_val=lambda_v)
  h_v_reversed = grl_v(h_v)
  adv_pred_v = Adversary_v(h_v_reversed)

  # Decoder hidden states
  grl_d = GradientReversalLayer(lambda_val=lambda_d)
  h_d_reversed = grl_d(h_d)
  adv_pred_d = Adversary_d(h_d_reversed)

  # Combined loss
  L_total = L_caption - lambda_v * L_adv_v - lambda_d * L_adv_d
  ```
- [ ] Hyperparameter search for λ_v and λ_d
  - Grid search: [0.3, 0.5, 0.7, 1.0]
  - Optimize for: min(adversary_acc) + max(BLEU)

**Adversary_d Architecture:**
```python
# Input: decoder hidden state (512-dim LSTM)
# Keras Sequential model
AdversaryNetwork_d:
  Dense(512 → 256) + BatchNormalization + LeakyReLU(alpha=0.2)
  Dense(256 → 256) + BatchNormalization + LeakyReLU(alpha=0.2)
  Dense(256 → 2, activation='softmax')  # race prediction
```

**Success Criteria:**
- Both Adversary_v and Adversary_d achieve <60% accuracy
- Caption BLEU maintained (≤5% drop from baseline)
- Demonstrates dual-location effectiveness

#### 2.2 Advanced Bias Metrics
**Files:** `src/evaluation/bias_metrics.py`

**Tasks:**
- [ ] Race-occupation association scores
  - Extract occupation words from captions
  - Compute P(occupation | race) vs dataset P(occupation)
  - Bias score = KL divergence or chi-squared test
- [ ] Bias amplification index (Δ)
  - Δ = (Model bias - Dataset bias) / Dataset bias
  - Goal: Δ → 0
- [ ] Unwanted racial descriptor frequency
  - Define list of racial descriptors (skin tone words)
  - Measure frequency in generated captions
  - Goal: minimize frequency while maintaining accuracy
- [ ] Model leakage (λ_M)
  - Train post-hoc adversary on frozen encoder/decoder features
  - Measure how well it predicts race
  - Goal: λ_M → 0.5 (random guessing)

**Implementation:**
```python
def compute_bias_amplification(model_predictions, dataset_labels):
    """
    Compute Δ for each attribute
    Δ = (λ_M - λ_D) / λ_D
    where λ_M = model leakage, λ_D = dataset leakage
    """
    # Dataset leakage: train adversary on ground truth labels
    # Model leakage: train adversary on model features
    pass
```

#### 2.3 Full Evaluation
**Files:** `src/evaluation/full_eval.py`

**Tasks:**
- [ ] Comprehensive comparison table
  - Baseline vs Encoder-only vs Dual-location
- [ ] Statistical significance tests (t-test, bootstrap)
- [ ] Qualitative analysis: example captions side-by-side
- [ ] Error analysis: where does debiasing hurt caption quality?

**Evaluation Protocol:**
```python
Models to Compare:
1. Baseline (no debiasing)
2. Encoder-only adversarial (λ_v only)
3. Dual-location adversarial (λ_v + λ_d)
4. Dual-location with optimized λ values

Metrics:
- Caption Quality: BLEU-4, ROUGE-L, CIDEr
- Bias: Adversary acc, Δ, race word frequency
- Fairness: Per-race BLEU, variance across races
```

---

### Phase 3: Research Extensions (Stretch Goal)

#### 3.1 Novel Bias Metrics for Generative Text
**Files:** `src/evaluation/generative_bias_metrics.py`

**Tasks:**
- [ ] Semantic similarity analysis
  - Generate captions with/without racial context
  - Compute sentence embeddings (BERT/SBERT)
  - Measure semantic drift caused by race
- [ ] Contextual embedding bias
  - Extract noun/verb/adjective distributions per race
  - Measure association strength using pointwise mutual information
- [ ] Stereotype detection
  - Define stereotype categories (occupations, activities, settings)
  - Measure differential usage across race groups

#### 3.2 Counterfactual Consistency
**Files:** `src/evaluation/counterfactual.py`, `src/data/counterfactual_dataset.py`

**Tasks:**
- [ ] Create counterfactual dataset
  - For each image, create synthetic version with different skin tone
  - Use image editing models (e.g., StyleGAN, facial attribute editing)
- [ ] Generate captions for original + counterfactual pairs
- [ ] Measure consistency metrics:
  - Object overlap: % of objects mentioned in both captions
  - Action consistency: same verbs used
  - Setting consistency: same location/context
  - Edit distance: Levenshtein distance between captions
- [ ] Bias-independent quality score:
  - Caption should change only for race-relevant attributes
  - Should remain identical for objects, actions, scenes

---

## Project Structure

```
project_root/
├── data/
│   ├── fiftyone/               # FiftyOne COCO data
│   ├── MASTER_TRAIN.csv        # Preprocessed training data
│   └── MASTER_VAL.csv          # Preprocessed validation data
├── src/
│   ├── data/
│   │   ├── dataset.py          # TensorFlow data pipeline
│   │   ├── data_generator.py   # Custom data generator with balancing
│   │   └── counterfactual_dataset.py  # Counterfactual pairs
│   ├── models/
│   │   ├── encoder.py          # Vision encoder (ResNet50)
│   │   ├── decoder.py          # Caption decoder (LSTM with attention)
│   │   ├── adversary.py        # Race classifier networks
│   │   ├── gradient_reversal.py # GradientReversalLayer (Keras)
│   │   ├── baseline_model.py   # No debiasing
│   │   ├── adv_model_encoder.py # Single adversary
│   │   └── adv_model_dual.py   # Dual adversaries
│   ├── training/
│   │   ├── baseline_train.py   # Train baseline
│   │   ├── adv_train_encoder.py # Single-location adversarial
│   │   ├── adv_train_dual.py   # Dual-location adversarial
│   │   └── losses.py           # Combined loss functions
│   ├── evaluation/
│   │   ├── caption_metrics.py  # BLEU, ROUGE, CIDEr
│   │   ├── bias_metrics.py     # Adversary probing, Δ
│   │   ├── generative_bias_metrics.py  # Novel metrics
│   │   ├── counterfactual.py   # Counterfactual analysis
│   │   └── full_eval.py        # Comprehensive evaluation
│   └── utils/
│       ├── config.py           # Hyperparameters
│       ├── logger.py           # TensorBoard logging
│       └── vocab.py            # Vocabulary builder
├── notebooks/
│   ├── data_exploration.ipynb  # EDA on dataset
│   └── results_analysis.ipynb  # Visualize results
├── checkpoints/                # Saved models (.h5 or SavedModel format)
├── logs/                       # TensorBoard logs
├── results/                    # Evaluation outputs
├── preprocess.py               # Data preprocessing
├── requirements.txt            # Dependencies
└── IMPLEMENTATION_PLAN.md      # This file
```

## Key Technical Details

### Gradient Reversal Layer (TensorFlow/Keras)
```python
import tensorflow as tf
from tensorflow import keras

@tf.custom_gradient
def gradient_reversal(x, lambda_val):
    """
    Gradient reversal function using TensorFlow custom gradient.
    Forward pass: identity
    Backward pass: negates and scales gradients by lambda_val
    """
    def grad(dy):
        return -lambda_val * dy, None
    return x, grad

class GradientReversalLayer(keras.layers.Layer):
    """
    Keras layer that reverses gradients during backpropagation.
    """
    def __init__(self, lambda_val=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_val = lambda_val

    def call(self, x):
        return gradient_reversal(x, self.lambda_val)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_val": self.lambda_val})
        return config
```

### Training Loop Pseudocode (TensorFlow)
```python
# Create optimizers
encoder_optimizer = tf.keras.optimizers.Adam(lr_encoder)
decoder_optimizer = tf.keras.optimizers.Adam(lr_decoder)
adversary_optimizer = tf.keras.optimizers.Adam(lr_adversary)

@tf.function
def train_step(images, captions, race_labels):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        encoder_features = encoder(images, training=True)  # [B, 2048]

        # Caption generation
        decoder_hiddens, predicted_captions = decoder(
            encoder_features, captions, training=True
        )
        caption_loss = caption_loss_fn(captions, predicted_captions)

        # Encoder adversary with gradient reversal
        reversed_features_v = gradient_reversal_layer_v(encoder_features)
        race_pred_v = adversary_v(reversed_features_v, training=True)
        adversary_loss_v = tf.keras.losses.sparse_categorical_crossentropy(
            race_labels, race_pred_v
        )

        # Decoder adversary with gradient reversal
        reversed_features_d = gradient_reversal_layer_d(decoder_hiddens)
        race_pred_d = adversary_d(reversed_features_d, training=True)
        adversary_loss_d = tf.keras.losses.sparse_categorical_crossentropy(
            race_labels, race_pred_d
        )

        # Combined loss for main model (encoder + decoder)
        main_loss = caption_loss - lambda_v * adversary_loss_v - lambda_d * adversary_loss_d

    # Compute gradients
    main_vars = encoder.trainable_variables + decoder.trainable_variables
    main_gradients = tape.gradient(main_loss, main_vars)

    # Apply gradients
    encoder_optimizer.apply_gradients(
        zip(main_gradients[:len(encoder.trainable_variables)],
            encoder.trainable_variables)
    )
    decoder_optimizer.apply_gradients(
        zip(main_gradients[len(encoder.trainable_variables):],
            decoder.trainable_variables)
    )

    del tape
    return caption_loss, adversary_loss_v, adversary_loss_d

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataset:
        images, captions, race_labels = batch
        caption_loss, adv_loss_v, adv_loss_d = train_step(
            images, captions, race_labels
        )
```

## Hyperparameters

### Model Architecture
```python
# Vision Encoder
encoder_type = 'resnet50'  # or 'clip-vit-b32'
encoder_dim = 2048  # ResNet50 output
finetune_encoder = False  # Start with frozen, finetune later

# Caption Decoder
decoder_type = 'lstm'  # or 'transformer'
embed_dim = 300
decoder_dim = 512
attention_dim = 512
dropout = 0.5

# Adversary Networks
adversary_v_dims = [2048, 512, 512, 512, 2]
adversary_d_dims = [512, 256, 256, 2]
```

### Training Hyperparameters
```python
# Learning rates
lr_encoder = 1e-4  # if finetuning
lr_decoder = 1e-3
lr_adversary = 1e-3

# Adversarial weights
lambda_v = 0.5  # encoder adversary (tune: 0.3-1.0)
lambda_d = 0.5  # decoder adversary (tune: 0.3-1.0)

# Training
batch_size = 32
num_epochs = 30
grad_clip = 5.0  # gradient clipping for stability
```

### Data Preprocessing (TensorFlow)
```python
# Image preprocessing function
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.random_crop(img, [224, 224, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32)
    # Normalize for ResNet50 (ImageNet preprocessing)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

# Caption tokenization
max_caption_length = 20
vocab_min_freq = 5
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    oov_token="<UNK>",
    filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
)
```

## Success Criteria

### Base Goal
- ✅ Single-location adversarial debiasing implemented
- ✅ Adversary_v accuracy drops compared to baseline (ideally <60%)
- ✅ Caption BLEU ≥ 95% of baseline

### Target Goal
- ✅ Dual-location gradient reversal fully implemented
- ✅ Demonstrates reasonable reduction in racial leakage (both adversaries <60% acc)
- ✅ Caption quality maintained (≤5% BLEU drop)

### Stretch Goal
- ✅ Novel bias evaluation metric for generative text developed
- ✅ Counterfactual consistency demonstrated across dataset
- ✅ Paper-ready results and visualizations

## Division of Labor

### Baseline Model
- Tommy
- Gun (Kanpat)

### Adversarial Model + Gradient Reversal
- Vignesh
- Gun (Kanpat)

### Training Pipeline with Debiasing
- Vignesh
- Joao

### Model Evaluation
- Tommy
- Joao

## References

1. **Balanced Datasets Are Not Enough** (Wang et al., ICCV 2019)
   - Repository: `Balanced-Datasets-Are-Not-Enough-master/`
   - Key file: `object_multilabel/adv/model.py` (gradient reversal)

2. **Understanding and Evaluating Racial Biases in Image Captioning** (Princeton)
   - Dataset: 28,315 images with skin color annotations
   - Paper: `Understanding and Evaluating Racial Biases in Image Captioning.pdf`

3. **Project Report**
   - File: `Intermediate Project Report.pdf`

## Timeline Estimate

- **Week 1-2:** Phase 1 (Baseline + Single-location adversarial)
- **Week 3:** Phase 2.1 (Dual-location implementation)
- **Week 4:** Phase 2.2-2.3 (Advanced metrics + full evaluation)
- **Week 5+:** Phase 3 (Stretch goals) + paper writing

## Notes

- Start simple: baseline → single adversary → dual adversaries
- Validate each component before moving to next phase
- Keep detailed logs of hyperparameter experiments
- Regularly evaluate on validation set to catch overfitting
- Prioritize caption quality - debiasing shouldn't destroy model utility
