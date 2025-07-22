# IndiCASA: Contrastive Learning for Text Encoders

## Overview

This project focuses on training various transformer-based text encoders using contrastive learning techniques. The primary goal is to fine-tune these encoders to produce embeddings that can effectively distinguish between similar (stereotypical) and dissimilar (anti-stereotypical) text pairs. The training process involves using different contrastive loss functions and evaluating the model's performance based on cosine similarity metrics between positive and negative pairs.

## Features

*   **Multiple Encoder Support:**
    *   `ModernBERT-base` (answerdotai/ModernBERT-base)
    *   `bert-base-cased` (google-bert/bert-base-cased)
    *   `all-MiniLM-L6-v2` (sentence-transformers/all-MiniLM-L6-v2)
*   **Variety of Contrastive Loss Functions:**
    *   `NTBXentLoss`: Normalized Temperature Binary Cross-Entropy Loss.
    *   `NTXentLoss`: Normalized Temperature-scaled Cross Entropy Loss.
    *   `PairLoss`: A loss function based on a similarity threshold for positive and negative pairs.
    *   `TripletLoss`: A loss function that encourages the distance between an anchor and a positive sample to be smaller than the distance between the anchor and a negative sample by a certain margin.
*   **Flexible Training Configuration:**
    *   Adjustable learning rates, loss function parameters (temperature/margin), number of epochs, and early stopping patience.
*   **Performance Tracking:**
    *   Logs training progress and evaluation metrics (cosine similarity for positive/negative pairs across different categories like caste, religion, etc.).
    *   Saves training history and results.

## File Structure

```
IndiCASA/
└── training_encoder/
    ├── encoders.py           # Defines the text encoder models
    ├── loss_fns.py           # Defines the contrastive loss functions
    ├── Train.py              # Main training script with Trainer class
    ├── train_encoder.py      # Script to configure and run training
    ├── README.md             # This documentation file
... (other project files)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Matrixmang0/IndiCASA.git
    cd training_encoder
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The primary dependencies are PyTorch and the Hugging Face Transformers library.
    ```bash
    pip install torch
    pip install transformers
    pip install scikit-learn
    pip install datasets
    pip install pandas
    pip install tqdm
    ```

## Usage

1.  **Prepare your dataset:**
    *   The `Train.py` script currently attempts to load data from `datasets/caste_data.pkl` and expects specific splits like `train_stereo`, `train_astereo`, etc. Ensure your data is preprocessed and structured accordingly. The `load_dataset` method in `Train.py` will need to be adapted if your dataset format or loading mechanism differs.

2.  **Configure Training Parameters:**
    *   Open `train_encoder.py`.
    *   Modify the `training_parameters` dictionary to set:
        *   `model_names`: Choose from `"all-MiniLM-L6-v2"`, `"google-bert/bert-base-cased"`, `"answerdotai/ModernBERT-base"`.
        *   `loss_param`: Values for temperature or margin depending on the loss function.
        *   `loss_param_type`: `"temperature"` or `"margin"`.
        *   `learning_rate`: The learning rate for the optimizer.
        *   `criterion`: Choose from `"NTXentLoss"`, `"NTBXentLoss"`, `"TripletLoss"`, `"PairLoss"`.
        *   `max_epochs`: Maximum number of training epochs.
        *   `early_stop_patience`: Number of epochs to wait for improvement before stopping.
        *   `pos_threshold`, `neg_threshold`: Thresholds used for evaluation or specific loss functions.
        *   `device`: `"cuda"` or `"cpu"`.

    Example configuration in `train_encoder.py`:
    ```python
    model_names = ["all-MiniLM-L6-v2", "google-bert/bert-base-cased", "answerdotai/ModernBERT-base"]
    loss_types = ["NTXentLoss", "NTBXentLoss", "TripletLoss", "PairLoss"]
    loss_params_types = ["temperature", "margin"]

    training_parameters = {
        model_names[2]: {  # Example: "answerdotai/ModernBERT-base"
            "loss_param": [0.1, 0.5],
            "loss_param_type": [loss_params_types[0]], # temperature
            "learning_rate": [5e-5],
            "criterion": [loss_types[0]], # NTXentLoss
            "max_epochs": [100],
            "early_stop_patience": [25],
            "pos_threshold": [90],
            "neg_threshold": [80],
            "device": ["cuda"],
        }
    }
    ```

3.  **Run Training:**
    Execute the `train_encoder.py` script from the `GitHub/training_encoder/` directory:
    ```bash
    python train_encoder.py
    ```

4.  **View Results:**
    *   Training logs will be printed to the console and saved in the `logs/` directory (e.g., `logs/training.log`).
    *   Training history (metrics per epoch) will be saved as JSON files in the `results/` directory.

## Code Modules

### `encoders.py`
Contains PyTorch `nn.Module` classes for different transformer encoders:
*   `ModernBertEncoder`: Wrapper for `answerdotai/ModernBERT-base`.
*   `BertEncoder`: Wrapper for `google-bert/bert-base-cased`.
*   `AllMiniLMEncoder`: Wrapper for `sentence-transformers/all-MiniLM-L6-v2`.
Each encoder class initializes the tokenizer and model from Hugging Face Transformers and provides a `forward` method to get sentence embeddings (specifically the CLS token embedding).

### `loss_fns.py`
Defines various contrastive loss functions as PyTorch `nn.Module` classes:
*   `NTBXentLoss`: Implements the Normalized Temperature Binary Cross-Entropy loss. It computes cosine similarity between all pairs in a combined batch of stereo and astereo embeddings and uses binary cross-entropy.
*   `NTXentLoss`: Implements the Normalized Temperature-scaled Cross Entropy loss (SimCLR loss). It aims to pull positive pairs (embeddings from two views of the same sample, or two similar samples) together and push all other pairs apart.
*   `PairLoss`: A custom loss that penalizes positive pairs if their similarity is below a threshold and negative pairs if their similarity is above another threshold.
*   `TripletLoss`: Implements the triplet margin loss. It considers an anchor, a positive sample, and a negative sample, and aims to make the anchor more similar to the positive than to the negative by a certain margin.

### `Train.py`
This script orchestrates the training process.
*   `setup_logger()`: Configures a logger for console and file output.
*   `Trainers` class:
    *   Initializes the model, loss function, optimizer, and loads the dataset.
    *   `load_dataset()`: Placeholder for loading and splitting stereotype/anti-stereotype data. **Needs to be adapted to your specific dataset.**
    *   `create_val_dataset()`: Creates validation pairs.
    *   `benchmark_cossim()`: Calculates initial cosine similarity scores on validation data.
    *   `get_init_metrics()`: Computes and logs initial metrics.
    *   `calc_val_loss()`: Calculates and logs validation loss and similarity metrics during training.
    *   `train_contrastive_model()`: The main training loop. It iterates through epochs, performs training steps, evaluates on validation data, and implements early stopping.
    *   `train()`: Orchestrates the overall training flow and saves results.
*   `ContrastiveTraining` class: A simple `nn.Module` wrapper around the chosen encoder model.
*   `train_encoder()`: A function (called from `train_encoder.py`) that iterates through training configurations and instantiates/runs the `Trainers` class.

### `train_encoder.py`
This is the main script to start the training process. It defines the hyperparameter grid (model names, loss functions, learning rates, etc.) and calls `train_encoder` from `Train.py` to run the experiments.

## Dependencies

*   `torch`: PyTorch library for tensor computations and neural networks.
*   `transformers`: Hugging Face Transformers library for pre-trained models and tokenizers.
*   `scikit-learn`: For `train_test_split`.
*   `datasets`: Hugging Face Datasets library (used for `load_from_disk`).
*   `tqdm`: For progress bars.
*   `logging`: Standard Python logging module.
*   `os`: Standard Python OS module.
*   `itertools`: For `combinations` and `product`.
*   `json`: For saving history.
*   `matplotlib` (implied by `plt.subplots` in commented-out code): For plotting, if uncommented.