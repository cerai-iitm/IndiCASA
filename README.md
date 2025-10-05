# IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context

## Overview

This project is dedicated to understanding, evaluating, and mitigating biases, particularly those related to social constructs like caste, religion, and gender, within language models, with a focus on the Indian context. It provides datasets, tools for bias evaluation, and methods for training text encoders using contrastive learning to better distinguish between stereotypical and anti-stereotypical content.

The primary goals include:
-   Providing robust datasets (`IndiCASA`) tailored for the Indian socio-cultural context.
-   Offering a framework (`BiasMetric`) to benchmark and evaluate fairness and bias in language models.
-   Developing techniques (`training_encoder`) to fine-tune text encoders to be more sensitive to nuanced stereotypical content.

## Repository Structure

The project is organized into several key directories:

```
IndiCASA/
├── BiasMetric/             # Tools and scripts for bias evaluation using IndiBias
│   ├── README.md
│   ├── Prepare_IndiBias.py
│   ├── BiasMetric.py
│   └── benchmark.py
├── IndiCASA_dataset/       # IndiCASA dataset, loading scripts, and documentation
│   ├── README.md
│   └── load_IndiCASA.ipynb
├── training_encoder/       # Scripts for training text encoders with contrastive learning
│   ├── README.md
│   ├── encoders.py
│   ├── loss_fns.py
│   ├── Train.py
│   └── train_encoder.py
├── requirements.txt         # Common dependencies for the project
└── README.md               # This main README file
```

## Key Components

### 1. Datasets

Two primary datasets are central to this project:

*   **IndiBias Dataset**:
    *   **Purpose**: Used for evaluating fairness and bias in language models. It contains sentences where specific tokens related to bias types (e.g., caste, religion) can be masked and predicted by models. Also, we used IndiBias to curate the IndiCASA dataset.
    *   **Details & Preparation**: Refer to the [BiasMetric/README.md](BiasMetric/README.md#getting-started) for instructions on preparing and using the IndiBias dataset. The preparation involves cleaning and aligning raw data, and creating masked versions for model evaluation.

*   **IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) Dataset**:
    *   **Purpose**: Provides contextually aligned sentences of stereotypical and anti-stereotypical sentences across various social dimensions relevant to India (caste, religion, gender, socioeconomic status, disability).
    *   **Details & Usage**: For information on the dataset schema, loading instructions, and research applications, please see the [IndiCASA_dataset/README.md](IndiCASA_dataset/README.md) and the [load_IndiCASA.ipynb](IndiCASA_dataset/load_IndiCASA.ipynb) notebook.

### 2. Bias Evaluation Framework (`BiasMetric`)

*   **Purpose**: This framework provides tools to quantitatively measure bias in language models. It includes functionalities for:
    *   Generating model responses to masked sentences from the IndiBias dataset.
    *   Extracting and validating these predictions.
    *   Computing sentence embeddings and cosine similarities.
    *   Calculating stereotyping scores and KL divergence metrics to assess bias.
*   **Setup & Usage**: Detailed instructions for setting up the environment and running the evaluation benchmarks can be found in [BiasMetric/README.md](BiasMetric/README.md).

### 3. Contrastive Encoder Training (`training_encoder`)

*   **Purpose**: This component focuses on fine-tuning various transformer-based text encoders using contrastive learning. The aim is to produce embeddings that can effectively differentiate between stereotypical and anti-stereotypical text pairs from datasets like IndiCASA.
*   **Features**:
    *   Supports multiple encoder architectures (e.g., BERT, MiniLM, ModernBERT).
    *   Implements various contrastive loss functions (e.g., NTXentLoss, TripletMarginLoss).
    *   Tracks performance using cosine similarity metrics.
*   **Setup & Usage**: For details on setting up the training environment, configuring parameters, and running the training scripts, please consult the [training_encoder/README.md](training_encoder/README.md).

## Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Matrixmang0/IndiCASA.git
    cd BiasMetric
    ```

2.  **Set up Python Environment**:
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Install common dependencies. Specific dependencies for each component are listed in their respective README files.
    ```bash
    pip install torch transformers datasets scikit-learn pandas tqdm
    # Refer to BiasMetric/requirements.txt for BiasMetric specific dependencies
    # Refer to training_encoder/README.md for training_encoder specific dependencies
    ```

4.  **Explore Components**:
    *   To work with datasets, see [IndiCASA_dataset/README.md](IndiCASA_dataset/README.md) and [BiasMetric/README.md](BiasMetric/README.md).
    *   To run bias evaluations, follow the instructions in [BiasMetric/README.md](BiasMetric/README.md).
    *   To train encoders, refer to [training_encoder/README.md](training_encoder/README.md).

## Research and Citation

This work aims to contribute to the growing body of research on fairness and safety in AI. If you use components of this project, please consider citing the relevant papers or the project itself.
The research related to the `BiasMetric` component can be found in the [BiasMetric/README.md](BiasMetric/README.md) file.

Please refer to the README files in subdirectories for more specific citation information if available.

## Ethical Considerations

This project involves datasets and analyses that deal with sensitive topics, including social stereotypes.
*   The inclusion of stereotypical content is for research purposes to understand and mitigate bias, and does not endorse these views.
*   Care should be taken when deploying models trained or evaluated on this data.
*   Researchers and developers are encouraged to implement appropriate bias mitigation strategies and be mindful of the potential societal impact of their work.

